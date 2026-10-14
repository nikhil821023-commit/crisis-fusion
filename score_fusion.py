"""
Phase 3: score fusion baseline.
Combines a trained text model checkpoint and a trained image model checkpoint
by averaging their predicted class probabilities.
"""

import argparse
import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-large",
    "bertweet": "vinai/bertweet-base",
}


def clean(t):
    t = re.sub(r"http\S+|@\w+", " ", str(t)).replace("#", "")
    t = t.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


class PairDS(Dataset):
    """Loads text and image from the same test rows."""

    def __init__(self, csv, tok):
        self.df = pd.read_csv(csv)
        self.tok = tok

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        e = self.tok(
            clean(r.text),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        img = self.tf(
            Image.open(r.image_path).convert("RGB")
        )

        return (
            e["input_ids"][0],
            e["attention_mask"][0],
            img,
            int(r.label),
        )


def load_text_model(name, n_classes, ckpt):
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT[name],
        num_labels=n_classes,
    )

    model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE)
    )

    return model.to(DEVICE).eval()


def load_image_model(name, n_classes, ckpt):
    if name != "vit":
        raise ValueError("Currently only ViT is supported for image fusion.")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=n_classes,
    )

    model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE)
    )

    return model.to(DEVICE).eval()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--text_model",
        required=True,
        choices=TEXT.keys(),
    )
    ap.add_argument(
        "--image_model",
        required=True,
        choices=["vit"],
    )
    ap.add_argument(
        "--data",
        default="data/processed",
    )
    ap.add_argument(
        "--bs",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="Weight given to text probabilities.",
    )

    args = ap.parse_args()

    print("=" * 70)
    print("PHASE 3 SCORE FUSION")
    print("=" * 70)
    print("Task       :", args.task)
    print("Text model :", args.text_model)
    print("Image model:", args.image_model)
    print("Device     :", DEVICE)
    print("Weight     :", args.weight)
    print("=" * 70)

    task_dir = os.path.join(args.data, args.task)

    label_map_path = os.path.join(
        task_dir,
        "label_map.json",
    )

    with open(label_map_path) as f:
        n_classes = len(json.load(f))

    print("Classes    :", n_classes)

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT[args.text_model]
    )

    test_csv = os.path.join(
        task_dir,
        "test.csv",
    )

    test_dataset = PairDS(
        test_csv,
        tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
    )

    text_ckpt = f"best_{args.task}_{args.text_model}.pt"
    image_ckpt = f"best_{args.task}_{args.image_model}.pt"

    print("Text checkpoint :", text_ckpt)
    print("Image checkpoint:", image_ckpt)

    if not os.path.exists(text_ckpt):
        raise FileNotFoundError(
            f"Missing text checkpoint: {text_ckpt}"
        )

    if not os.path.exists(image_ckpt):
        raise FileNotFoundError(
            f"Missing image checkpoint: {image_ckpt}"
        )

    print("Loading text model...")
    text_model = load_text_model(
        args.text_model,
        n_classes,
        text_ckpt,
    )

    print("Loading image model...")
    image_model = load_image_model(
        args.image_model,
        n_classes,
        image_ckpt,
    )

    print("Running fusion...")

    ys = []
    predictions = []

    with torch.no_grad():
        for ids, mask, img, y in test_loader:

            text_output = text_model(
                input_ids=ids.to(DEVICE),
                attention_mask=mask.to(DEVICE),
            )

            image_output = image_model(
                pixel_values=img.to(DEVICE),
            )

            text_prob = torch.softmax(
                text_output.logits,
                dim=1,
            )

            image_prob = torch.softmax(
                image_output.logits,
                dim=1,
            )

            fused_prob = (
                args.weight * text_prob
                + (1 - args.weight) * image_prob
            )

            predictions.extend(
                fused_prob.argmax(dim=1)
                .cpu()
                .tolist()
            )

            ys.extend(y.tolist())

    acc = accuracy_score(
        ys,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            ys,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    result = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print("=" * 70)
    print("SCORE FUSION RESULT")
    print("=" * 70)
    print(result)
    print("=" * 70)

    row = pd.DataFrame([
        {
            "task": args.task,
            "model": f"score_fusion({args.text_model}+{args.image_model})",
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ])

    row.to_csv(
        "results.csv",
        mode="a",
        header=not os.path.exists("results.csv"),
        index=False,
    )

    print("Saved result to results.csv")


if __name__ == "__main__":
    main()
