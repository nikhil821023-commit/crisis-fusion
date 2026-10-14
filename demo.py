
import argparse
import json
import os

import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer

from fusion_model import CrossModalFusionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_CONFIG = {
    "task1a": {
        "ckpt": "best_task1a_fusion.pt",
        "model_name": "roberta-large",
        "num_classes": 2,
        "labels": None,
    },
    "task1b": {
        "ckpt": "best_task1b_fusion.pt",
        "model_name": "roberta-large",
        "num_classes": 5,
        "labels": None,
    },
    "task2": {
        "ckpt": "best_task2_fusion.pt",
        "model_name": "vinai/bertweet-base",
        "num_classes": 8,
        "labels": None,
    },
    "task3": {
        "ckpt": "best_task3_fusion.pt",
        "model_name": "roberta-large",
        "num_classes": 3,
        "labels": None,
    },
}


def load_labels(task):
    path = f"data/processed/{task}/label_map.json"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # label_map is usually {"label": index}
        if all(str(v).isdigit() for v in label_map.values()):
            labels = [None] * len(label_map)
            for label, idx in label_map.items():
                labels[int(idx)] = str(label)
            return labels

        # fallback if map is index -> label
        return [str(label_map[str(i)]) for i in range(len(label_map))]

    return [f"Class {i}" for i in range(TASK_CONFIG[task]["num_classes"])]


def load_model(task, checkpoint):
    config = TASK_CONFIG[task]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    model = CrossModalFusionModel(
        num_classes=config["num_classes"]
    )

    state = torch.load(checkpoint, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def predict(task, tweet, image, checkpoint):
    if image is None:
        return {"Error": 1.0}

    if tweet is None:
        tweet = ""

    tokenizer, model = load_model(task, checkpoint)

    encoded = tokenizer(
        tweet,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    image = image.convert("RGB")

    # Match the ViT preprocessing used by the project.
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image_tensor,
        )

        probabilities = torch.softmax(logits, dim=1)[0]

    labels = load_labels(task)

    result = {}

    for i, probability in enumerate(probabilities):
        label = labels[i] if i < len(labels) else f"Class {i}"
        result[label] = float(probability)

    return result


def create_demo(task, checkpoint):

    labels = load_labels(task)

    def predict_wrapper(tweet, image):
        return predict(
            task=task,
            tweet=tweet,
            image=image,
            checkpoint=checkpoint,
        )

    demo = gr.Interface(
        fn=predict_wrapper,
        inputs=[
            gr.Textbox(
                label="Tweet / Text",
                placeholder="Enter the disaster-related tweet here..."
            ),
            gr.Image(
                type="pil",
                label="Disaster Image"
            ),
        ],
        outputs=gr.Label(
            num_top_classes=len(labels),
            label="Prediction Confidence"
        ),
        title=f"CrisisMMD Multimodal Demo — {task}",
        description=(
            "Upload a disaster image and enter the associated tweet. "
            "The multimodal model combines textual and visual information "
            "to produce class predictions."
        ),
        examples=None,
    )

    return demo


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["task1a", "task1b", "task2", "task3"],
        default="task1a",
    )

    parser.add_argument(
        "--ckpt",
        default=None,
    )

    parser.add_argument(
        "--share",
        action="store_true",
    )

    args = parser.parse_args()

    config = TASK_CONFIG[args.task]

    checkpoint = args.ckpt or config["ckpt"]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Please provide the correct checkpoint using --ckpt."
        )

    print(f"Task: {args.task}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {DEVICE}")

    demo = create_demo(args.task, checkpoint)

    demo.launch(
        share=args.share,
        debug=True,
    )
