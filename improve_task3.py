
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from transformers import ViTForImageClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    matthews_corrcoef,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")


# ============================================================
# DATASET
# ============================================================

class Task3Dataset(Dataset):

    def __init__(
        self,
        csv_path,
        train=False,
        augment=False,
        limit=None
    ):

        self.df = pd.read_csv(csv_path)

        if limit is not None:
            self.df = self.df.head(limit).reset_index(drop=True)

        self.train = train

        # ----------------------------------------------------
        # BASE TRANSFORMS
        # ----------------------------------------------------

        aug = []

        if train:
            aug.append(
                transforms.RandomHorizontalFlip()
            )

            if augment:
                aug.extend([
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05
                    )
                ])

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

        image = Image.open(
            r.image_path
        ).convert("RGB")

        return (
            self.tf(image),
            int(r.label)
        )


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):

    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, num_classes):

    model.eval()

    ys = []
    ps = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(
                pixel_values=images
            ).logits

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .tolist()
            )

            ps.extend(predictions)
            ys.extend(labels.tolist())

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            ys,
            ps,
            labels=list(range(num_classes)),
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            ys,
            ps,
            average="weighted",
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            ys,
            ps,
            average="macro",
            zero_division=0
        )
    )

    result = {
        "acc": accuracy_score(ys, ps),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy_score(
            ys,
            ps
        ),
        "mcc": matthews_corrcoef(
            ys,
            ps
        )
    }

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    for i in range(num_classes):

        print(
            f"Class {i}: "
            f"Precision={precision[i]:.4f} "
            f"Recall={recall[i]:.4f} "
            f"F1={f1[i]:.4f} "
            f"Support={support[i]}"
        )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    for k, v in result.items():

        print(
            f"{k:20s}: {v:.4f}"
        )

    return result


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for step, (images, labels) in enumerate(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            pixel_values=images
        ).logits

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:

            print(
                f"  step {step + 1}/{len(loader)} "
                f"loss={loss.item():.4f}"
            )

    return total_loss / max(1, len(loader))


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="data/processed"
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    ap.add_argument(
        "--bs",
        type=int,
        default=16
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None
    )

    ap.add_argument(
        "--class_weights",
        action="store_true"
    )

    ap.add_argument(
        "--augment",
        action="store_true"
    )

    ap.add_argument(
        "--oversample",
        action="store_true"
    )

    a = ap.parse_args()


    # ========================================================
    # SEED
    # ========================================================

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)


    # ========================================================
    # PATHS
    # ========================================================

    train_csv = os.path.join(
        a.data,
        "task3",
        "train.csv"
    )

    dev_csv = os.path.join(
        a.data,
        "task3",
        "dev.csv"
    )

    test_csv = os.path.join(
        a.data,
        "task3",
        "test.csv"
    )

    label_map_path = os.path.join(
        a.data,
        "task3",
        "label_map.json"
    )


    with open(
        label_map_path,
        encoding="utf-8"
    ) as f:

        label_map = json.load(f)


    num_classes = len(label_map)


    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = Task3Dataset(
        train_csv,
        train=True,
        augment=a.augment,
        limit=a.limit
    )

    dev_ds = Task3Dataset(
        dev_csv,
        train=False,
        augment=False
    )

    test_ds = Task3Dataset(
        test_csv,
        train=False,
        augment=False
    )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    train_labels = (
        train_ds.df["label"]
        .astype(int)
        .values
    )

    counts = np.bincount(
        train_labels,
        minlength=num_classes
    )


    print()
    print("=" * 70)
    print("PHASE 6 - TASK 3")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Epochs       : {a.epochs}")
    print(f"Batch size   : {a.bs}")
    print(f"Learning rate: {a.lr}")
    print(f"Seed         : {a.seed}")
    print(f"Limit        : {a.limit}")
    print(f"Class weights: {a.class_weights}")
    print(f"Augmentation: {a.augment}")
    print(f"Oversample   : {a.oversample}")
    print(f"Classes      : {num_classes}")

    print()
    print("Training class counts:")

    for i, count in enumerate(counts):

        print(
            f"  class {i}: {count}"
        )


    # ========================================================
    # DATALOADER
    # ========================================================

    sampler = None

    if a.oversample:

        weights = np.zeros(
            num_classes,
            dtype=np.float64
        )

        for i in range(num_classes):

            if counts[i] > 0:
                weights[i] = 1.0 / counts[i]

        sample_weights = np.array([
            weights[label]
            for label in train_labels
        ])

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(
                sample_weights
            ),
            num_samples=len(sample_weights),
            replacement=True
        )


    train_loader = DataLoader(
        train_ds,
        batch_size=a.bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=a.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=a.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Loading ViT...")

    model = build_model(
        num_classes
    ).to(DEVICE)


    # ========================================================
    # LOSS
    # ========================================================

    if a.class_weights:

        class_weight_values = (
            len(train_labels)
            /
            (
                num_classes
                * np.maximum(counts, 1)
            )
        )

        class_weights = torch.tensor(
            class_weight_values,
            dtype=torch.float32,
            device=DEVICE
        )

        print()
        print(
            "Using class weights:",
            class_weights.detach()
            .cpu()
            .numpy()
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    else:

        criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr
    )


    # ========================================================
    # TRAIN
    # ========================================================

    best_macro_f1 = -1.0

    best_path = (
        "best_task3_improved.pt"
    )

    for epoch in range(
        1,
        a.epochs + 1
    ):

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch}/{a.epochs}"
        )
        print("=" * 70)

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        print(
            f"Training loss: {loss:.4f}"
        )

        print()
        print("Validation:")

        val_result = evaluate(
            model,
            dev_loader,
            num_classes
        )

        if (
            val_result["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_result["macro_f1"]
            )

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                f"Saved best model: "
                f"{best_path}"
            )


    # ========================================================
    # TEST
    # ========================================================

    if os.path.exists(best_path):

        model.load_state_dict(
            torch.load(
                best_path,
                map_location=DEVICE
            )
        )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    result = evaluate(
        model,
        test_loader,
        num_classes
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_row = {
        "task": "task3",
        "model": "vit",
        "seed": a.seed,
        "epochs": a.epochs,
        "batch_size": a.bs,
        "lr": a.lr,
        "class_weights": a.class_weights,
        "augment": a.augment,
        "oversample": a.oversample,
        **result
    }


    output = (
        "task3_improvement_results.csv"
    )

    new_df = pd.DataFrame(
        [result_row]
    )

    if os.path.exists(output):

        old_df = pd.read_csv(
            output
        )

        final_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    else:

        final_df = new_df


    final_df.to_csv(
        output,
        index=False
    )


    print()
    print(
        f"Saved results to {output}"
    )


if __name__ == "__main__":
    main()
