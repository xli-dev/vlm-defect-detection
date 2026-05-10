"""
Experiments 2 & 5: Cross-Domain Linear Probe
Freeze CNN or VLM backbone trained on source domain (carpet),
train a logistic regression on target domain train set,
evaluate on target domain test set.

Supports few-shot mode (--n_shot) for the most discriminating comparison.

Usage:
    python src/linear_probe.py --backbone cnn --source carpet --target leather
    python src/linear_probe.py --backbone cnn --source carpet --target leather --n_shot 5
    python src/linear_probe.py --backbone vlm --source carpet --target leather --n_shot 5
    python src/linear_probe.py --backbone vlm --source carpet --target leather \
        --adapter results/vlm_carpet_adapter --n_shot 5
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import models
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from dataset import MVTecDataset, BINARY_CLASSES, default_transform

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_ID = "google/paligemma-3b-mix-224"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Few-shot sampler ──────────────────────────────────────────────────────────

def few_shot_sample(samples, n_shot, seed=42):
    """Sample n_shot images per class for few-shot linear probe training."""
    rng = random.Random(seed)
    class_to_samples = defaultdict(list)
    for path, label in samples:
        class_to_samples[label].append((path, label))

    selected = []
    for label, items in class_to_samples.items():
        items = sorted(items)
        rng.shuffle(items)
        selected.extend(items[:n_shot])

    n_classes = len(class_to_samples)
    print(f"  Few-shot: {n_shot} samples × {n_classes} classes "
          f"= {len(selected)} train images")
    return selected


# ── CNN feature extractor ─────────────────────────────────────────────────────

def load_cnn_extractor(source, device):
    weights_path = RESULTS_DIR / f"cnn_{source}_backbone.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"CNN backbone not found at {weights_path}.\n"
            f"Run: python src/train_cnn.py --source {source}"
        )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Remove FC — use 2048-dim penultimate features
    extractor = nn.Sequential(*list(model.children())[:-1])
    return extractor.to(device).eval()


@torch.no_grad()
def extract_cnn_features(extractor, samples, device):
    transform = default_transform()
    features, labels = [], []
    for path, label in tqdm(samples, desc="CNN features", leave=False):
        img  = Image.open(path).convert("RGB")
        x    = transform(img).unsqueeze(0).to(device)
        feat = extractor(x).squeeze().cpu().numpy()
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)


# ── VLM feature extractor ─────────────────────────────────────────────────────

def load_vlm_extractor(adapter_path, device):
    print(f"Loading VLM: {MODEL_ID}")
    processor  = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        label = "fine-tuned"
    else:
        print("Using zero-shot VLM features")
        model = base_model
        label = "zero-shot"

    return model.to(device).eval(), processor, label


@torch.no_grad()
def extract_vlm_features(model, processor, samples, device):
    prompt   = "Is this industrial surface image defective or normal?"
    features, labels = [], []
    for path, label in tqdm(samples, desc="VLM features", leave=False):
        img    = Image.open(path).convert("RGB")
        inputs = processor(images=img, text=prompt,
                           return_tensors="pt").to(device)
        # Navigate to vision_tower regardless of PEFT wrapping
        # Check class name to distinguish PEFT from base model
        if "Peft" in type(model).__name__:
            # PEFT: PeftModelForCausalLM.base_model.model = PaliGemmaForConditionalGeneration
            paligemma = model.base_model.model
        else:
            # Base: PaliGemmaForConditionalGeneration
            paligemma = model
        vision_out = paligemma.model.vision_tower(pixel_values=inputs["pixel_values"])
        pooled = (vision_out.last_hidden_state
                  .mean(dim=1).squeeze().cpu().float().numpy())
        features.append(pooled)
        labels.append(label)
    return np.array(features), np.array(labels)


# ── Linear probe ──────────────────────────────────────────────────────────────

def run_linear_probe(train_feats, train_labels, test_feats, test_labels):
    scaler      = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats  = scaler.transform(test_feats)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(train_feats, train_labels)

    preds  = clf.predict(test_feats)
    probs  = clf.predict_proba(test_feats)[:, 1]
    acc    = (preds == test_labels).mean()
    auc    = roc_auc_score(test_labels, probs) if len(set(test_labels)) > 1 else 0.0
    report = classification_report(test_labels, preds,
                                   target_names=BINARY_CLASSES,
                                   output_dict=True, zero_division=0)
    return acc, auc, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"Cross-domain: {args.source} → {args.target}")

    # Load target domain data
    target_train = MVTecDataset(args.target, split="train")
    target_test  = MVTecDataset(args.target, split="test")

    train_samples = target_train.samples
    if args.n_shot:
        print(f"Few-shot mode: {args.n_shot} samples/class")
        train_samples = few_shot_sample(train_samples, args.n_shot)

    n_shot_label = f"{args.n_shot}shot" if args.n_shot else "full"

    # Extract features
    if args.backbone == "cnn":
        extractor = load_cnn_extractor(args.source, device)
        print("\nExtracting CNN features...")
        train_feats, train_labels = extract_cnn_features(
            extractor, train_samples, device)
        test_feats, test_labels = extract_cnn_features(
            extractor, target_test.samples, device)
        backbone_label = f"ResNet-50 (trained on {args.source})"
        out_file = (RESULTS_DIR /
                    f"exp2_cnn_{args.source}_{args.target}_{n_shot_label}.json")
    else:
        model, processor, vlm_label = load_vlm_extractor(args.adapter, device)
        print("\nExtracting VLM features...")
        train_feats, train_labels = extract_vlm_features(
            model, processor, train_samples, device)
        test_feats, test_labels = extract_vlm_features(
            model, processor, target_test.samples, device)
        if vlm_label == "fine-tuned":
            backbone_label = f"PaliGemma-3B (fine-tuned on {args.source})"
        else:
            backbone_label = "PaliGemma-3B (pre-trained, no fine-tuning)"
        out_file = (RESULTS_DIR /
                    f"exp5_vlm_{vlm_label}_{args.source}_{args.target}_{n_shot_label}.json")

    print(f"\nRunning {n_shot_label} linear probe...")
    acc, auc, report = run_linear_probe(
        train_feats, train_labels, test_feats, test_labels)

    print(f"\n{'='*50}")
    print(f"Cross-domain linear probe: {args.source} → {args.target}")
    print(f"Backbone: {backbone_label}")
    print(f"Probe training: {n_shot_label}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"{'='*50}")

    with open(out_file, "w") as f:
        json.dump({
            "backbone": backbone_label,
            "source": args.source,
            "target": args.target,
            "n_shot": args.n_shot,
            "accuracy": acc,
            "auc": auc,
            "report": report,
        }, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["cnn", "vlm"])
    parser.add_argument("--source",   default="carpet",
                        choices=["carpet", "grid", "leather", "tile", "wood"])
    parser.add_argument("--target",   required=True,
                        choices=["carpet", "grid", "leather", "tile", "wood"])
    parser.add_argument("--adapter",  default=None,
                        help="Path to VLM LoRA adapter")
    parser.add_argument("--n_shot",   type=int, default=None,
                        help="N samples/class for few-shot probe (default: all)")
    main(parser.parse_args())
