
import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import fusion_model as fm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


TASK_INFO = {
    "task1a": {
        "num_classes": 2,
        "test_csv": "data/processed/task1a/test.csv",
    },
    "task1b": {
        "num_classes": 5,
        "test_csv": "data/processed/task1b/test.csv",
    },
    "task2": {
        "num_classes": 2,
        "test_csv": "data/processed/task2/test.csv",
    },
    "task3": {
        "num_classes": 4,
        "test_csv": "data/processed/task3/test.csv",
    },
}


def forward_with_ablation(model, batch, ablation):

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    image = batch["image"].to(DEVICE)

    text_output = model.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    text_tokens = model.text_projection(
        text_output.last_hidden_state
    )

    image_output = model.image_encoder(
        pixel_values=image
    )

    image_tokens = model.image_projection(
        image_output.last_hidden_state
    )

    if ablation == "text_only":
        image_tokens = torch.zeros_like(image_tokens)

    elif ablation == "image_only":
        text_tokens = torch.zeros_like(text_tokens)

    if ablation != "no_cross_modal":

        text_attended, _ = model.text_to_image_attention(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
        )

        text_tokens = model.text_norm(
            text_tokens + text_attended
        )

        image_attended, _ = model.image_to_text_attention(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
        )

        image_tokens = model.image_norm(
            image_tokens + image_attended
        )

    multimodal_tokens = torch.cat(
        [text_tokens, image_tokens],
        dim=1
    )

    if ablation != "no_self_attn":

        attended_tokens, _ = model.self_attention(
            query=multimodal_tokens,
            key=multimodal_tokens,
            value=multimodal_tokens,
        )

        multimodal_tokens = model.self_norm(
            multimodal_tokens + attended_tokens
        )

    refined = model.ffn(multimodal_tokens)

    multimodal_tokens = model.ffn_norm(
        multimodal_tokens + refined
    )

    text_count = text_tokens.size(1)

    fused_text = multimodal_tokens[:, :text_count, :]
    fused_image = multimodal_tokens[:, text_count:, :]

    text_pool = fused_text.mean(dim=1)
    image_pool = fused_image.mean(dim=1)

    fused = torch.cat(
        [text_pool, image_pool],
        dim=1
    )

    return model.classifier(fused)


def calculate_metrics(labels, predictions, probabilities, num_classes):

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        labels, predictions
    )

    metrics["precision"] = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    metrics["recall"] = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    metrics["f1"] = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    metrics["mcc"] = matthews_corrcoef(
        labels,
        predictions
    )

    metrics["kappa"] = cohen_kappa_score(
        labels,
        predictions
    )

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        labels,
        predictions
    )

    try:

        probabilities = np.asarray(probabilities)

        if num_classes == 2:

            metrics["auc_roc"] = roc_auc_score(
                labels,
                probabilities[:, 1]
            )

        else:

            metrics["auc_roc"] = roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

    except Exception:

        metrics["auc_roc"] = float("nan")

    return metrics


def save_confusion_matrix(
    labels,
    predictions,
    task,
    ablation,
    num_classes
):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes))
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        f"{task} - {ablation}\\nConfusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))

    for i in range(num_classes):

        for j in range(num_classes):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    path = os.path.join(
        FIGURE_DIR,
        f"{task}_{ablation}_confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return path


def save_result(task, ablation, metrics):

    path = "ablation_results.csv"

    exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "task",
                "ablation",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "kappa",
                "balanced_accuracy",
                "auc_roc",
            ])

        writer.writerow([
            task,
            ablation,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["mcc"],
            metrics["kappa"],
            metrics["balanced_accuracy"],
            metrics["auc_roc"],
        ])


def run_evaluation(task, ablation="full"):

    info = TASK_INFO[task]

    num_classes = info["num_classes"]
    test_csv = info["test_csv"]

    checkpoint_path = f"best_{task}_fusion.pt"

    print("=" * 70)
    print("PHASE 5 — EVALUATION")
    print("=" * 70)

    print("Task:", task)
    print("Ablation:", ablation)
    print("Device:", DEVICE)
    print("Checkpoint:", checkpoint_path)
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        fm.TEXT_MODEL
    )

    test_dataset = fm.CrisisDataset(
        test_csv,
        tokenizer,
        max_length=128,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Test samples:", len(test_dataset))

    model = fm.CrossModalFusionModel(
        num_classes=num_classes,
        fusion_dim=512,
        num_heads=8,
        dropout=0.1
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded.")
    print("No training will be performed.")

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in test_loader:

            if ablation == "full":

                logits = fm.forward_batch(
                    model,
                    batch
                )

            else:

                logits = forward_with_ablation(
                    model,
                    batch,
                    ablation
                )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            labels = batch["label"].to(DEVICE)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
        num_classes
    )

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        f"Accuracy          : {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision         : {metrics['precision']:.4f}"
    )

    print(
        f"Recall            : {metrics['recall']:.4f}"
    )

    print(
        f"F1                : {metrics['f1']:.4f}"
    )

    print(
        f"MCC               : {metrics['mcc']:.4f}"
    )

    print(
        f"Cohen's Kappa     : {metrics['kappa']:.4f}"
    )

    print(
        f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"AUC-ROC           : {metrics['auc_roc']:.4f}"
    )

    print()
    print("PER-CLASS REPORT")
    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(num_classes)),
            zero_division=0,
            digits=4
        )
    )

    figure_path = save_confusion_matrix(
        all_labels,
        all_predictions,
        task,
        ablation,
        num_classes
    )

    print(
        "Confusion matrix saved:",
        figure_path
    )

    save_result(
        task,
        ablation,
        metrics
    )

    print(
        "Result saved to ablation_results.csv"
    )

    print("=" * 70)

    return metrics


def show_summary():

    path = "ablation_results.csv"

    if not os.path.exists(path):

        print("No ablation_results.csv found.")
        return

    df = pd.read_csv(path)

    print()
    print("=" * 100)
    print("PHASE 5 SUMMARY")
    print("=" * 100)

    print(
        df.to_string(index=False)
    )

    print("=" * 100)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=[
            "task1a",
            "task1b",
            "task2",
            "task3"
        ]
    )

    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "no_cross_modal",
            "no_self_attn",
            "text_only",
            "image_only"
        ],
        default="full"
    )

    parser.add_argument(
        "--summary",
        action="store_true"
    )

    args = parser.parse_args()

    if args.summary:

        show_summary()
        return

    if args.task is None:

        parser.error(
            "--task is required unless --summary is used"
        )

    run_evaluation(
        args.task,
        args.ablation
    )


if __name__ == "__main__":
    main()
