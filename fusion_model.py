
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTModel,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL = "roberta-large"
IMAGE_MODEL = "google/vit-base-patch16-224-in21k"

TASKS = {
    "task1a": "data/processed/task1a",
    "task1b": "data/processed/task1b",
    "task2": "data/processed/task2",
    "task3": "data/processed/task3",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean(text):
    text = str(text)

    # Remove URLs
    import re
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove # symbol but keep the word
    text = text.replace("#", "")

    # Keep basic readable characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# DATASET
# ============================================================

class CrisisDataset(Dataset):

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=128,
        train=False,
        limit=None,
    ):

        self.df = pd.read_csv(csv_path)

        if limit is not None:
            self.df = self.df.iloc[:limit].copy()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = clean(row["text"])

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        image_path = str(row["image_path"]).replace("\\", "/")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# CROSS-MODAL FUSION MODEL
# ============================================================

class CrossModalFusionModel(nn.Module):

    def __init__(
        self,
        num_classes,
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Pretrained encoders
        # ----------------------------------------------------

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL
        )

        self.image_encoder = ViTModel.from_pretrained(
            IMAGE_MODEL
        )

        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size

        # RoBERTa-large = 1024
        # ViT-base = 768

        # ----------------------------------------------------
        # Project both modalities into same dimension
        # ----------------------------------------------------

        self.text_projection = nn.Linear(
            text_dim,
            fusion_dim
        )

        self.image_projection = nn.Linear(
            image_dim,
            fusion_dim
        )

        # ----------------------------------------------------
        # Cross-modal attention
        #
        # Text queries image
        # Image queries text
        # ----------------------------------------------------

        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Layer normalization after cross attention
        # ----------------------------------------------------

        self.text_norm = nn.LayerNorm(fusion_dim)
        self.image_norm = nn.LayerNorm(fusion_dim)

        # ----------------------------------------------------
        # Self-attention over combined multimodal tokens
        # ----------------------------------------------------

        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(fusion_dim)

        # ----------------------------------------------------
        # Feed-forward refinement
        # ----------------------------------------------------

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        self.ffn_norm = nn.LayerNorm(fusion_dim)

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        image,
    ):

        # ====================================================
        # TEXT ENCODER
        # ====================================================

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_tokens = text_output.last_hidden_state

        # [B, text_length, 1024]
        # ↓
        # [B, text_length, 512]

        text_tokens = self.text_projection(text_tokens)

        # ====================================================
        # IMAGE ENCODER
        # ====================================================

        image_output = self.image_encoder(
            pixel_values=image
        )

        image_tokens = image_output.last_hidden_state

        # [B, 197, 768]
        # ↓
        # [B, 197, 512]

        image_tokens = self.image_projection(image_tokens)

        # ====================================================
        # TEXT -> IMAGE CROSS ATTENTION
        #
        # Text asks:
        # "Which visual regions are relevant to this text?"
        # ====================================================

        text_attended, _ = self.text_to_image_attention(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
        )

        text_tokens = self.text_norm(
            text_tokens + text_attended
        )

        # ====================================================
        # IMAGE -> TEXT CROSS ATTENTION
        #
        # Image asks:
        # "Which textual information is relevant to this image?"
        # ====================================================

        image_attended, _ = self.image_to_text_attention(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
        )

        image_tokens = self.image_norm(
            image_tokens + image_attended
        )

        # ====================================================
        # COMBINE MODALITIES
        # ====================================================

        multimodal_tokens = torch.cat(
            [
                text_tokens,
                image_tokens,
            ],
            dim=1,
        )

        # ====================================================
        # SELF ATTENTION
        # ====================================================

        attended_tokens, _ = self.self_attention(
            query=multimodal_tokens,
            key=multimodal_tokens,
            value=multimodal_tokens,
        )

        multimodal_tokens = self.self_norm(
            multimodal_tokens + attended_tokens
        )

        # ====================================================
        # FEED FORWARD
        # ====================================================

        refined = self.ffn(multimodal_tokens)

        multimodal_tokens = self.ffn_norm(
            multimodal_tokens + refined
        )

        # ====================================================
        # GLOBAL POOLING
        #
        # First half = text tokens
        # Second half = image tokens
        # ====================================================

        text_count = text_tokens.size(1)

        fused_text = multimodal_tokens[:, :text_count, :]
        fused_image = multimodal_tokens[:, text_count:, :]

        text_pool = fused_text.mean(dim=1)
        image_pool = fused_image.mean(dim=1)

        # [B, 512] + [B, 512]
        fused = torch.cat(
            [
                text_pool,
                image_pool,
            ],
            dim=1,
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        logits = self.classifier(fused)

        return logits


# ============================================================
# FORWARD HELPER
# ============================================================

def forward_batch(model, batch):

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    image = batch["image"].to(DEVICE)

    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image=image,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:

        logits = forward_batch(model, batch)

        preds = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        labels = batch["label"].numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=["task1a", "task1b", "task2", "task3"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--bs",
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--freeze",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    print()
    print("=" * 70)
    print("PHASE 4 PROPOSED CROSS-MODAL FUSION")
    print("=" * 70)
    print(f"Task       : {args.task}")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.bs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed       : {args.seed}")
    print(f"Data limit : {args.limit}")
    print(f"Freeze     : {args.freeze}")
    print("=" * 70)

    task_dir = TASKS[args.task]

    # --------------------------------------------------------
    # Load label map
    # --------------------------------------------------------

    with open(
        os.path.join(task_dir, "label_map.json"),
        "r",
        encoding="utf-8",
    ) as f:

        label_map = json.load(f)

    num_classes = len(label_map)

    print(f"Classes    : {num_classes}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds = CrisisDataset(
        os.path.join(task_dir, "train.csv"),
        tokenizer,
        train=True,
        limit=args.limit,
    )

    dev_ds = CrisisDataset(
        os.path.join(task_dir, "dev.csv"),
        tokenizer,
        train=False,
    )

    test_ds = CrisisDataset(
        os.path.join(task_dir, "test.csv"),
        tokenizer,
        train=False,
    )

    print(f"train: {len(train_ds)} samples")
    print(f"dev:   {len(dev_ds)} samples")
    print(f"test:  {len(test_ds)} samples")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print("Building proposed fusion model...")

    model = CrossModalFusionModel(
        num_classes=num_classes,
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Freeze pretrained encoders if requested
    # --------------------------------------------------------

    if args.freeze:

        for p in model.text_encoder.parameters():
            p.requires_grad = False

        for p in model.image_encoder.parameters():
            p.requires_grad = False

        print()
        print(
            "Backbones frozen — training fusion layers only"
        )

    else:

        print()
        print(
            "Full fine-tuning enabled"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    print()
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in trainable_params
        ),
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0

    checkpoint = (
        f"best_{args.task}_fusion.pt"
    )

    print()
    print("Starting training...")

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            logits = forward_batch(
                model,
                batch
            )

            labels = batch["label"].to(
                DEVICE
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(train_loader)
        )

        dev_result = evaluate(
            model,
            dev_loader
        )

        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={avg_loss:.3f} "
            f"dev acc={dev_result['acc']:.4f} "
            f"f1={dev_result['f1']:.4f}"
        )

        if dev_result["f1"] > best_f1:

            best_f1 = dev_result["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task": args.task,
                    "num_classes": num_classes,
                    "fusion_dim": 512,
                    "num_heads": 8,
                },
                checkpoint,
            )

            print(
                f"  saved best model -> {checkpoint}"
            )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print("Loading best model...")

    saved = torch.load(
        checkpoint,
        map_location=DEVICE,
    )

    model.load_state_dict(
        saved["model_state_dict"]
    )

    print("Running test...")

    result = evaluate(
        model,
        test_loader
    )

    print()
    print("=" * 70)
    print("PROPOSED FUSION TEST RESULT")
    print("=" * 70)
    print(result)
    print("=" * 70)

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result_row = pd.DataFrame(
        [
            {
                "task": args.task,
                "model": "proposed_fusion",
                "seed": args.seed,
                "limit": (
                    "" if args.limit is None
                    else args.limit
                ),
                "acc": result["acc"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        ]
    )

    results_file = "results.csv"

    if os.path.exists(results_file):

        existing = pd.read_csv(
            results_file
        )

        # Make sure schema remains consistent
        columns = [
            "task",
            "model",
            "seed",
            "limit",
            "acc",
            "precision",
            "recall",
            "f1",
        ]

        existing = existing[columns]

        result_row = result_row[columns]

        combined = pd.concat(
            [
                existing,
                result_row,
            ],
            ignore_index=True,
        )

    else:

        combined = result_row

    combined.to_csv(
        results_file,
        index=False,
    )

    print()
    print(
        f"Saved result to {results_file}"
    )


if __name__ == "__main__":
    main()
