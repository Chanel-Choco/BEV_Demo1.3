"""
make_samples.py -- builds the sample gallery (samples/ folder) from your thesis test split.

Run it once, on a machine that has the Notebook 1 outputs (test_split.csv and the resized_dataset folder):

    python make_samples.py --splits path/to/test_split.csv --images path/to/resized_dataset

It runs both models on the test split and copies a handful of images into samples/:
  one confident correct AI image, one confident correct human image, the most confident false positive
  (human made, called AI), the most confident missed AI image, plus one correct AI image from each of
  MidJourney and Stable Diffusion when those sources are present. It also writes samples/samples.json with
  titles and the known labels. The images are copied unchanged, so the app skips the JPEG re-save for them
  (preprocessed=true).

Check the licence of any image you plan to show in public.
"""
import argparse
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from inference import DEVICE, EVAL_TRANSFORM, load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="path to test_split.csv (columns rel_path,label,source)")
    ap.add_argument("--images", required=True, help="path to the resized_dataset folder")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models, missing = load_models(args.weights)
    if missing or not models:
        raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))

    df = pd.read_csv(args.splits)
    probs = []
    with torch.no_grad():
        for i, rel in enumerate(df["rel_path"]):
            img = Image.open(os.path.join(args.images, rel)).convert("RGB")
            x = EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            ps = [torch.sigmoid(m(x)[0]).item() for m in models.values()]
            probs.append(float(np.mean(ps)))
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(df)} images scored")
    df["p_ai"] = probs
    df["pred"] = (df["p_ai"] >= 0.5).astype(int)

    rng = random.Random(args.seed)
    picks = []  # (row, key, title, note)

    def choose(mask, key, title, note, hardest_high=None):
        sub = df[mask]
        if sub.empty:
            return
        if hardest_high is True:
            row = sub.sort_values("p_ai", ascending=False).iloc[0]
        elif hardest_high is False:
            row = sub.sort_values("p_ai", ascending=True).iloc[0]
        else:
            row = sub.iloc[rng.randrange(len(sub))]
        picks.append((row, key, title, note))

    ai, human = df["label"] == 1, df["label"] == 0
    choose(ai & (df["p_ai"] > 0.9), "ai_correct", "AI, correct", "The ensemble gets this one right.")
    choose(human & (df["p_ai"] < 0.1), "human_correct", "Human, correct", "The ensemble gets this one right.")
    choose(human & (df["pred"] == 1), "human_false_positive", "Human, called AI",
           "A false positive, the most confident one in the test split. Look at where Grad-CAM points.",
           hardest_high=True)
    choose(ai & (df["pred"] == 0), "ai_missed", "AI, missed",
           "A missed AI image, the most confident miss in the test split.", hardest_high=False)
    if "source" in df.columns:
        for src, title in (("MidJourney", "MidJourney, correct"), ("StableDiffusion", "Stable Diffusion, correct")):
            choose(ai & (df["source"].astype(str).str.replace(" ", "").str.lower() == src.lower())
                   & (df["p_ai"] > 0.8), f"{src.lower()}_correct", title, "The ensemble gets this one right.")

    os.makedirs(args.out, exist_ok=True)
    meta = {}
    for row, key, title, note in picks:
        src_path = os.path.join(args.images, row["rel_path"])
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        name = f"{len(meta) + 1:02d}_{key}{ext}"
        shutil.copyfile(src_path, os.path.join(args.out, name))
        meta[name] = {"title": title, "label": "AI" if int(row["label"]) == 1 else "Human",
                      "note": f"{note} (ensemble P(AI) {row['p_ai']:.1%}, source {row.get('source', 'n/a')}.)",
                      "preprocessed": True}
    with open(os.path.join(args.out, "samples.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {len(meta)} samples to {args.out}")


if __name__ == "__main__":
    main()
