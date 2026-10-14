
import argparse
import json
import os
import re

import numpy as np
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

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


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

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=n_classes,
    )

    model.load_state_dict(
        torch.load(ckpt, map_location=DEVICE)
    )

    return model.to(DEVICE).eval()


def evaluate(task, text_model_name, weight, bs):

    task_dir = os.path.join(
        "data",
        "processed",
        task,
    )

    with open(
        os.path.join(task_dir, "label_map.json")
    ) as f:
        n_classes = len(json.load(f))

    print("=" * 70)
    print("PHASE 5 SCORE FUSION EVALUATION")
    print("=" * 70)
    print("Task       :", task)
    print("Text model :", text_model_name)
    print("Image model: vit")
    print("Weight     :", weight)
    print("Classes    :", n_classes)
    print("Device     :", DEVICE)
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT[text_model_name]
    )

    dataset = PairDS(
        os.path.join(task_dir, "test.csv"),
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
    )

    text_ckpt = f"best_{task}_{text_model_name}.pt"
    image_ckpt = f"best_{task}_vit.pt"

    print("Loading text model...")
    text_model = load_text_model(
        text_model_name,
        n_classes,
        text_ckpt,
    )

    print("Loading image model...")
    image_model = load_image_model(
        "vit",
        n_classes,
        image_ckpt,
    )

    ys = []
    predictions = []
    probabilities = []

    print("Running evaluation...")

    with torch.no_grad():

        for ids, mask, img, y in loader:

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
                weight * text_prob
                + (1 - weight) * image_prob
            )

            predictions.extend(
                fused_prob.argmax(dim=1)
                .cpu()
                .tolist()
            )

            probabilities.extend(
                fused_prob.cpu().numpy()
            )

            ys.extend(y.tolist())

    ys = np.array(ys)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # ------------------------------------------------------------
    # Main metrics
    # ------------------------------------------------------------

    acc = accuracy_score(ys, predictions)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            ys,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            ys,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    mcc = matthews_corrcoef(
        ys,
        predictions,
    )

    kappa = cohen_kappa_score(
        ys,
        predictions,
    )

    balanced_acc = balanced_accuracy_score(
        ys,
        predictions,
    )

    # ------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------

    try:

        if n_classes == 2:

            auc = roc_auc_score(
                ys,
                probabilities[:, 1],
            )

        else:

            auc = roc_auc_score(
                ys,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )

    except ValueError:

        auc = float("nan")

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        ys,
        predictions,
        labels=list(range(n_classes)),
    )

    # ------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------

    report = classification_report(
        ys,
        predictions,
        labels=list(range(n_classes)),
        output_dict=True,
        zero_division=0,
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro Precision   : {precision_macro:.4f}")
    print(f"Macro Recall      : {recall_macro:.4f}")
    print(f"Macro F1          : {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall   : {recall_weighted:.4f}")
    print(f"Weighted F1      : {f1_weighted:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Cohen Kappa       : {kappa:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(cm)

    print()
    print("=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    for i in range(n_classes):

        row = report[str(i)]

        print(
            f"Class {i}: "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1-score']:.4f}, "
            f"Support={int(row['support'])}"
        )

    # ------------------------------------------------------------
    # Save confusion matrix
    # ------------------------------------------------------------

    os.makedirs("figures", exist_ok=True)

    cm_path = os.path.join(
        "figures",
        f"{task}_score_fusion_confusion_matrix.csv",
    )

    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(n_classes)],
        columns=[f"pred_{i}" for i in range(n_classes)],
    ).to_csv(cm_path)

    # ------------------------------------------------------------
    # Save detailed result
    # ------------------------------------------------------------

    result = {
        "task": task,
        "model": f"score_fusion({text_model_name}+vit)",
        "weight": weight,
        "acc": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "mcc": mcc,
        "kappa": kappa,
        "balanced_accuracy": balanced_acc,
        "roc_auc": auc,
    }

    result_path = os.path.join(
        "figures",
        f"{task}_score_fusion_metrics.json",
    )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("Saved:")
    print(cm_path)
    print(result_path)

    return result


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--task", required=True)

    ap.add_argument(
        "--text_model",
        required=True,
        choices=TEXT.keys(),
    )

    ap.add_argument(
        "--weight",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--bs",
        type=int,
        default=16,
    )

    args = ap.parse_args()

    evaluate(
        args.task,
        args.text_model,
        args.weight,
        args.bs,
    )


if __name__ == "__main__":
    main()
