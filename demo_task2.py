
import argparse
import json
import os
import re

import gradio as gr
import torch

from PIL import Image
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL = "vinai/bertweet-base"
IMAGE_MODEL = "google/vit-base-patch16-224-in21k"

TEXT_CKPT = "best_task2_bertweet.pt"
IMAGE_CKPT = "best_task2_vit.pt"

WEIGHT = 0.5

LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
]


def clean(text):
    text = str(text)

    # Same cleaning used by score_fusion.py
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = text.replace("#", "")
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("=" * 70)
print("TASK 2 HUMANITARIAN SCORE-FUSION DEMO")
print("=" * 70)
print("Device:", DEVICE)
print("Text model:", TEXT_MODEL)
print("Image model:", IMAGE_MODEL)
print("Text checkpoint:", TEXT_CKPT)
print("Image checkpoint:", IMAGE_CKPT)
print("Fusion weight:", WEIGHT)
print("=" * 70)


# ------------------------------------------------------------
# Check checkpoints
# ------------------------------------------------------------

if not os.path.exists(TEXT_CKPT):
    raise FileNotFoundError(f"Missing checkpoint: {TEXT_CKPT}")

if not os.path.exists(IMAGE_CKPT):
    raise FileNotFoundError(f"Missing checkpoint: {IMAGE_CKPT}")


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

print("Loading BERTweet tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)


# ------------------------------------------------------------
# Load text model
# ------------------------------------------------------------

print("Loading BERTweet model...")

text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL,
    num_labels=len(LABELS),
)

text_model.load_state_dict(
    torch.load(TEXT_CKPT, map_location=DEVICE)
)

text_model.to(DEVICE)
text_model.eval()


# ------------------------------------------------------------
# Load image model
# ------------------------------------------------------------

print("Loading ViT model...")

image_model = ViTForImageClassification.from_pretrained(
    IMAGE_MODEL,
    num_labels=len(LABELS),
)

image_model.load_state_dict(
    torch.load(IMAGE_CKPT, map_location=DEVICE)
)

image_model.to(DEVICE)
image_model.eval()


# ------------------------------------------------------------
# Image preprocessing
# Same preprocessing used in score_fusion.py
# ------------------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict(tweet, image):

    if image is None:
        return {"Please upload an image": 1.0}

    if tweet is None:
        tweet = ""

    tweet = clean(tweet)

    # ----------------------------
    # Text
    # ----------------------------

    encoded = tokenizer(
        tweet,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # ----------------------------
    # Image
    # ----------------------------

    image = image.convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # ----------------------------
    # Model inference
    # ----------------------------

    with torch.no_grad():

        text_output = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_output = image_model(
            pixel_values=image_tensor,
        )

        text_prob = torch.softmax(
            text_output.logits,
            dim=1,
        )

        image_prob = torch.softmax(
            image_output.logits,
            dim=1,
        )

        # Exact 50/50 probability fusion
        fused_prob = (
            WEIGHT * text_prob
            + (1 - WEIGHT) * image_prob
        )

        fused_prob = fused_prob[0]

    # ----------------------------
    # Build result
    # ----------------------------

    result = {}

    for i, label in enumerate(LABELS):
        result[label] = float(fused_prob[i])

    return result


# ------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------

demo = gr.Interface(
    fn=predict,

    inputs=[
        gr.Textbox(
            label="Disaster Tweet / Text",
            placeholder="Enter the disaster-related tweet...",
            lines=5,
        ),

        gr.Image(
            type="pil",
            label="Disaster Image",
        ),
    ],

    outputs=gr.Label(
        num_top_classes=8,
        label="Humanitarian Classification"
    ),

    title="CrisisMMD — Humanitarian Information Detection",

    description=(
        "Task 2: The model combines the disaster tweet and image "
        "using 50/50 score fusion of BERTweet and ViT. "
        "It predicts the type of humanitarian information present "
        "in the disaster-related post."
    ),
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--share",
        action="store_true",
    )

    args = parser.parse_args()

    demo.launch(
        share=args.share,
        debug=True,
    )
