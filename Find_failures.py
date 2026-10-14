
"""
Find real misclassified examples from the test set — i.e. cases the model actually
got wrong. Much more useful for a report's "failure analysis" section than a guess.

Run:
  python find_failures.py --task task1a
"""
import argparse, json, os, re
import pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, ViTModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean(t):
    t = re.sub(r"http\S+|@\w+", " ", str(t)).replace("#", "")
    t = t.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


class FusionModel(nn.Module):
    def __init__(self, n_classes, d=128):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained("roberta-large")
        self.img_enc = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.Et_conv, self.Et_lin = nn.Conv1d(128, 1, 1), nn.Linear(1024, d)
        self.Ei_conv, self.Ei_lin = nn.Conv1d(197, 1, 1), nn.Linear(768, d)
        self.Es = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.dec = nn.Sequential(nn.Linear(4 * d, d), nn.Tanh(), nn.Linear(d, n_classes))

    def forward(self, ids, mask, img):
        with torch.no_grad():
            ht = self.text_enc(input_ids=ids, attention_mask=mask).last_hidden_state
            hi = self.img_enc(pixel_values=img).last_hidden_state
        t = torch.tanh(self.Et_lin(torch.tanh(self.Et_conv(ht)).squeeze(1)))
        i = torch.tanh(self.Ei_lin(torch.tanh(self.Ei_conv(hi)).squeeze(1)))
        t_s, i_s = torch.tanh(self.Es(t)), torch.tanh(self.Es(i))
        xm = torch.stack([t, i, t_s, i_s], dim=1)
        xa = self.norm2(self.attn(self.norm1(xm)))
        return self.dec(xa.flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=5, help="how many failure cases to show")
    a = ap.parse_args()

    d = os.path.join(a.data, a.task)
    label_map = json.load(open(os.path.join(d, "label_map.json")))
    id_to_label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    tok = AutoTokenizer.from_pretrained("roberta-large")
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = FusionModel(n_classes).to(DEVICE).eval()
    ckpt = a.ckpt or f"best_{a.task}_fusion.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    df = pd.read_csv(os.path.join(d, "test.csv"))
    failures = []
    print(f"Scanning {len(df)} test samples for misclassifications...")

    with torch.no_grad():
        for idx, r in df.iterrows():
            e = tok(clean(r.text), padding="max_length", truncation=True, max_length=128,
                   return_tensors="pt")
            img = tf(Image.open(r.image_path).convert("RGB")).unsqueeze(0)
            out = model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE), img.to(DEVICE))
            prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred = prob.argmax()
            if pred != r.label:
                failures.append(dict(
                    text=r.text, image_path=r.image_path,
                    true_label=id_to_label[r.label], pred_label=id_to_label[pred],
                    confidence=float(prob[pred]),
                ))
            if idx % 200 == 0:
                print(f"  ...{idx}/{len(df)} scanned, {len(failures)} failures so far")

    print(f"\nTotal misclassified: {len(failures)} / {len(df)} ({100*len(failures)/len(df):.1f}%)")
    print(f"\nTop {a.n} failure cases (use these in the demo to show a wrong prediction):\n")
    for f in failures[:a.n]:
        print("=" * 70)
        print("IMAGE PATH:", f["image_path"])
        print("TWEET TEXT:", f["text"])
        print(f"TRUE LABEL: {f['true_label']}   |   MODEL PREDICTED: {f['pred_label']} "
             f"({f['confidence']*100:.0f}% confidence)")

    pd.DataFrame(failures).to_csv(f"failures_{a.task}.csv", index=False)
    print(f"\nSaved all {len(failures)} failures to failures_{a.task}.csv")


if __name__ == "__main__":
    main()
