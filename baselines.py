
"""
Phase 2: unimodal baselines on the CSVs made in Phase 1.

Smoke test:
  python baselines.py --task task1a --model bert --epochs 1 --limit 100

CPU prototype:
  python baselines.py --task task1a --model bert --freeze --limit 2000 --epochs 2

Full run:
  python baselines.py --task task1a --model bert

Models:
  text  = bert | roberta | bertweet
  image = resnet50 | densenet121 | vit

Tasks:
  task1a | task1b | task2 | task3

Print comparison table:
  python baselines.py --table
"""

import argparse
import json
import os
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")


# ============================================================
# MODELS
# ============================================================

TEXT = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-large",
    "bertweet": "vinai/bertweet-base",
}

LR = {
    "bert": 2e-5,
    "roberta": 1e-5,
    "bertweet": 2e-5,
    "resnet50": 1e-4,
    "densenet121": 1e-4,
    "vit": 2e-5,
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean(t):
    t = re.sub(r"http\S+|@\w+", " ", str(t))
    t = t.replace("#", "")
    t = t.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# DATASET
# ============================================================

class DS(Dataset):

    def __init__(self, csv, model, train, tok=None, limit=None):

        self.df = pd.read_csv(csv)
        self.model = model
        self.tok = tok

        # ----------------------------------------------------
        # Optional dataset limit
        # ----------------------------------------------------

        if limit is not None:
            self.df = self.df.head(limit).reset_index(drop=True)

        # ----------------------------------------------------
        # Image transforms
        # ----------------------------------------------------

        aug = [
            transforms.RandomHorizontalFlip()
        ] if train else []

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        r = self.df.iloc[i]

        # ----------------------------------------------------
        # TEXT MODEL
        # ----------------------------------------------------

        if self.model in TEXT:

            e = self.tok(
                clean(r.text),
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            return {
                "input_ids": e["input_ids"][0],
                "attention_mask": e["attention_mask"][0]
            }, int(r.label)

        # ----------------------------------------------------
        # IMAGE MODEL
        # ----------------------------------------------------

        image = Image.open(
            r.image_path
        ).convert("RGB")

        return {
            "pixel_values": self.tf(image)
        }, int(r.label)


# ============================================================
# BUILD MODEL
# ============================================================

def build(model, n):

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if model in TEXT:

        return AutoModelForSequenceClassification.from_pretrained(
            TEXT[model],
            num_labels=n
        )

    # --------------------------------------------------------
    # ViT
    # --------------------------------------------------------

    if model == "vit":

        return ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=n
        )

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    if model == "resnet50":

        m = models.resnet50(
            weights="IMAGENET1K_V1"
        )

        m.fc = nn.Linear(
            m.fc.in_features,
            n
        )

        return m

    # --------------------------------------------------------
    # DenseNet
    # --------------------------------------------------------

    m = models.densenet121(
        weights="IMAGENET1K_V1"
    )

    m.classifier = nn.Linear(
        m.classifier.in_features,
        n
    )

    return m


# ============================================================
# FORWARD PASS
# ============================================================

def fwd(m, x):

    x = {
        k: v.to(DEVICE)
        for k, v in x.items()
    }

    if hasattr(m, "config"):
        return m(**x).logits

    return m(x["pixel_values"])


# ============================================================
# EVALUATION
# ============================================================

def evaluate(m, loader):

    m.eval()

    ys = []
    ps = []

    with torch.no_grad():

        for x, y in loader:

            predictions = (
                fwd(m, x)
                .argmax(1)
                .cpu()
                .tolist()
            )

            ps += predictions
            ys += y.tolist()

    p, r, f, _ = precision_recall_fscore_support(
        ys,
        ps,
        average="weighted",
        zero_division=0
    )

    return {
        "acc": accuracy_score(ys, ps),
        "precision": p,
        "recall": r,
        "f1": f
    }


# ============================================================
# RESULTS TABLE
# ============================================================

def show_table():

    if not os.path.exists("results.csv"):

        print("No results.csv found yet.")
        return

    df = pd.read_csv("results.csv")

    print()
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    table = (
        df
        .groupby(["task", "model"])[
            ["acc", "precision", "recall", "f1"]
        ]
        .mean()
        .round(4)
    )

    print(table)

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    # ========================================================
    # BASIC ARGUMENTS
    # ========================================================

    ap.add_argument(
        "--task",
        type=str
    )

    ap.add_argument(
        "--model",
        type=str
    )

    # ========================================================
    # DATA
    # ========================================================

    ap.add_argument(
        "--data",
        default="data/processed"
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None
    )

    # ========================================================
    # TRAINING
    # ========================================================

    ap.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    ap.add_argument(
        "--bs",
        type=int,
        default=16
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=None
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=3
    )

    # ========================================================
    # CPU-FRIENDLY MODE
    # ========================================================

    ap.add_argument(
        "--freeze",
        action="store_true"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # ========================================================
    # TABLE
    # ========================================================

    ap.add_argument(
        "--table",
        action="store_true"
    )

    # ========================================================
    # PARSE
    # ========================================================

    a = ap.parse_args()

    # ========================================================
    # TABLE MODE
    # ========================================================

    if a.table:
        return show_table()

    # ========================================================
    # VALIDATION
    # ========================================================

    valid_tasks = [
        "task1a",
        "task1b",
        "task2",
        "task3"
    ]

    valid_models = [
        "bert",
        "roberta",
        "bertweet",
        "resnet50",
        "densenet121",
        "vit"
    ]

    if a.task not in valid_tasks:

        raise ValueError(
            "Invalid task. Use task1a, task1b, task2 or task3."
        )

    if a.model not in valid_models:

        raise ValueError(
            "Invalid model."
        )

    # ========================================================
    # SEED
    # ========================================================

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    # ========================================================
    # DATA PATH
    # ========================================================

    d = os.path.join(
        a.data,
        a.task
    )

    label_map_path = os.path.join(
        d,
        "label_map.json"
    )

    with open(
        label_map_path,
        encoding="utf-8"
    ) as f:

        n = len(json.load(f))

    # ========================================================
    # INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 2 BASELINE")
    print("=" * 70)

    print(f"Task       : {a.task}")
    print(f"Model      : {a.model}")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {a.epochs}")
    print(f"Batch size : {a.bs}")
    print(f"Seed       : {a.seed}")
    print(f"Data limit : {a.limit}")
    print(f"Freeze     : {a.freeze}")
    print(f"Classes    : {n}")

    print("=" * 70)
    print()

    # ========================================================
    # TOKENIZER
    # ========================================================

    tok = None

    if a.model in TEXT:

        print("Loading tokenizer...")

        tok = AutoTokenizer.from_pretrained(
            TEXT[a.model]
        )

    # ========================================================
    # DATALOADERS
    # ========================================================

    def make_loader(split, shuffle):

        dataset = DS(
            os.path.join(
                d,
                f"{split}.csv"
            ),
            a.model,
            split == "train",
            tok,
            a.limit
        )

        print(
            f"{split}: {len(dataset)} samples"
        )

        return DataLoader(
            dataset,
            batch_size=a.bs,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(DEVICE == "cuda")
        )

    tr = make_loader(
        "train",
        True
    )

    dv = make_loader(
        "dev",
        False
    )

    te = make_loader(
        "test",
        False
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Building model...")

    model = build(
        a.model,
        n
    ).to(DEVICE)

    # ========================================================
    # FREEZE BACKBONE
    # ========================================================

    if a.freeze:

        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False

        # ----------------------------------------------------
        # Hugging Face text models
        # ----------------------------------------------------

        if a.model in TEXT:

            for p in model.classifier.parameters():
                p.requires_grad = True

        # ----------------------------------------------------
        # ViT
        # ----------------------------------------------------

        elif a.model == "vit":

            for p in model.classifier.parameters():
                p.requires_grad = True

        # ----------------------------------------------------
        # ResNet
        # ----------------------------------------------------

        elif a.model == "resnet50":

            for p in model.fc.parameters():
                p.requires_grad = True

        # ----------------------------------------------------
        # DenseNet
        # ----------------------------------------------------

        elif a.model == "densenet121":

            for p in model.classifier.parameters():
                p.requires_grad = True

        print()
        print(
            "Backbone frozen — training classifier head only"
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        trainable_params,
        lr=a.lr or LR[a.model],
        weight_decay=0.01
    )

    # ========================================================
    # AMP
    # ========================================================

    use_amp = DEVICE == "cuda"

    scaler = torch.amp.GradScaler(
        DEVICE,
        enabled=use_amp
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    ckpt = (
        f"best_{a.task}_{a.model}.pt"
    )

    # Add different name for frozen experiment
    if a.freeze:
        ckpt = (
            f"best_{a.task}_{a.model}_frozen.pt"
        )

    best = -1
    bad = 0

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("Starting training...")
    print()

    for ep in range(a.epochs):

        model.train()

        running_loss = 0.0
        batches = 0

        for x, y in tr:

            opt.zero_grad()

            with torch.autocast(
                device_type=DEVICE,
                enabled=use_amp
            ):

                logits = fwd(
                    model,
                    x
                )

                loss = nn.functional.cross_entropy(
                    logits,
                    y.to(DEVICE)
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                opt
            )

            scaler.update()

            running_loss += loss.item()
            batches += 1

        avg_loss = (
            running_loss / batches
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        m = evaluate(
            model,
            dv
        )

        print(
            f"epoch {ep + 1}/{a.epochs} "
            f"loss={avg_loss:.3f} "
            f"dev acc={m['acc']:.4f} "
            f"f1={m['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if m["f1"] > best:

            best = m["f1"]
            bad = 0

            torch.save(
                model.state_dict(),
                ckpt
            )

            print(
                f"  saved best model -> {ckpt}"
            )

        else:

            bad += 1

            if bad >= a.patience:

                print(
                    "early stopping"
                )

                break

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("Loading best model...")

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=DEVICE
        )
    )

    print("Running test...")

    res = evaluate(
        model,
        te
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("TEST:", res)
    print("=" * 70)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    row = pd.DataFrame([
        dict(
            task=a.task,
            model=a.model,
            seed=a.seed,
            limit=a.limit,
            freeze=a.freeze,
            **res
        )
    ])

    row.to_csv(
        "results.csv",
        mode="a",
        header=not os.path.exists("results.csv"),
        index=False
    )

    print()
    print("Saved result to results.csv")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

