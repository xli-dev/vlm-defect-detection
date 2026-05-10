"""
Experiments 3 & 4: VLM Zero-Shot Evaluation

Usage:
    python src/eval_vlm_zeroshot.py --category carpet
    python src/eval_vlm_zeroshot.py --category leather
    python src/eval_vlm_zeroshot.py --category tile
    python src/eval_vlm_zeroshot.py --category wood
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from dataset import MVTecDataset, BINARY_CLASSES, TEXTURE_CATEGORIES

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_ID = "google/paligemma-3b-mix-224"

# Simple VQA-style prompt that PaliGemma-mix handles well
PROMPT = "<image> Does this surface have any visible defects such as scratches, holes, or cracks? Answer yes or no."


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_zeroshot(model, processor, dataset, device):
    all_preds, all_labels = [], []

    for path, label in tqdm(dataset.samples, desc="Zero-shot eval"):
        img    = Image.open(path).convert("RGB")
        inputs = processor(images=img, text=PROMPT,
                           return_tensors="pt").to(device)
        output   = model.generate(**inputs, max_new_tokens=5)
        response = processor.decode(output[0], skip_special_tokens=True).strip().lower()

        # Model echoes prompt then appends answer after newline — take last word
        last_word = response.strip().split()[-1] if response.strip() else "normal"

        # Debug: print first 5 responses
        if len(all_preds) < 5:
            print(f"  [{'defect' if label==1 else 'good':>6}] → {last_word!r}")

        if any(w in last_word for w in ("yes", "defective", "defect", "abnormal", "damaged")):
            pred = 1
        else:
            pred = 0

        all_preds.append(pred)
        all_labels.append(label)

    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    auc    = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(all_labels, all_preds,
                                   target_names=BINARY_CLASSES,
                                   output_dict=True, zero_division=0)
    return acc, auc, report


def main(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"Category: {args.category}")

    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).to(device).eval()

    dataset = MVTecDataset(args.category, split="test")
    acc, auc, report = evaluate_zeroshot(model, processor, dataset, device)

    print(f"\nZero-shot on {args.category}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")

    out_path = RESULTS_DIR / f"exp_vlm_zeroshot_{args.category}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "PaliGemma-3B-mix (zero-shot)",
            "category": args.category,
            "accuracy": acc,
            "auc": auc,
            "report": report,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                        choices=TEXTURE_CATEGORIES)
    main(parser.parse_args())
