"""
Experiment 6: Feature Space Visualization (t-SNE + Silhouette Scores)
Compares CNN vs VLM features across source and target domains.

Key metrics:
  Per-domain silhouette: how well good/defective separate WITHIN each domain
  Global silhouette by domain: how much features cluster BY domain

Usage:
    python src/visualize_features.py --source carpet --targets carpet leather tile wood
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from torchvision import models
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

from dataset import MVTecDataset, TEXTURE_CATEGORIES, default_transform, MVTEC_DIR

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_ID    = "google/paligemma-3b-mix-224"
N_PER_CLASS = 30


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_category(category, n_per_class, split="test"):
    ds  = MVTecDataset(category, split=split)
    good   = [(p, l) for p, l in ds.samples if l == 0]
    defect = [(p, l) for p, l in ds.samples if l == 1]
    rng    = random.Random(42)
    rng.shuffle(good)
    rng.shuffle(defect)
    chosen  = good[:n_per_class] + defect[:n_per_class]
    paths   = [p for p, _ in chosen]
    labels  = [l for _, l in chosen]
    domains = [category] * len(chosen)
    return paths, labels, domains


@torch.no_grad()
def extract_cnn_features(paths, source, device):
    weights = RESULTS_DIR / f"cnn_{source}_backbone.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Backbone not found: {weights}. Run train_cnn.py first.")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights, map_location=device))
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    transform = default_transform()
    features  = []
    for p in tqdm(paths, desc="CNN features", leave=False):
        img  = Image.open(p).convert("RGB")
        x    = transform(img).unsqueeze(0).to(device)
        feat = extractor(x).squeeze().cpu().numpy()
        features.append(feat)
    return np.array(features)


@torch.no_grad()
def extract_vlm_features(paths, device, adapter_path=None):
    processor  = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        label = "fine-tuned"
    else:
        model = base_model.to(device)
        label = "zero-shot"
    model.eval()
    prompt   = "Is this industrial surface image defective or normal?"
    features = []
    for p in tqdm(paths, desc=f"VLM features ({label})", leave=False):
        img    = Image.open(p).convert("RGB")
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        if "Peft" in type(model).__name__:
            paligemma = model.base_model.model
        else:
            paligemma = model
        vision_out = paligemma.model.vision_tower(pixel_values=inputs["pixel_values"])
        pooled = vision_out.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy()
        features.append(pooled)
    return np.array(features), label


def compute_and_plot(features, labels, domains, model_name, save_prefix):
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(features)

    label_arr  = np.array(labels)
    domain_arr = np.array(domains)

    # Global silhouette
    sil_label  = silhouette_score(features, label_arr)  if len(set(labels))  > 1 else 0
    sil_domain = silhouette_score(features, domain_arr) if len(set(domains)) > 1 else 0

    # Per-domain silhouette — measures within-domain good/defective separation
    unique_domains = sorted(set(domains))
    per_domain_sil = {}
    for dom in unique_domains:
        mask = domain_arr == dom
        dom_feats  = features[mask]
        dom_labels = label_arr[mask]
        if len(set(dom_labels)) > 1:
            per_domain_sil[dom] = silhouette_score(dom_feats, dom_labels)
        else:
            per_domain_sil[dom] = float("nan")

    mean_per_domain = float(np.nanmean(list(per_domain_sil.values())))

    print(f"  Global silhouette by label:  {sil_label:.4f}")
    print(f"  Global silhouette by domain: {sil_domain:.4f}")
    print(f"  Per-domain silhouette (good/defect within each domain):")
    for dom, score in per_domain_sil.items():
        print(f"    {dom:<10}: {score:.4f}")
    print(f"  Mean per-domain silhouette:  {mean_per_domain:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{model_name}\n"
        f"Mean per-domain silhouette={mean_per_domain:.3f}  |  "
        f"Global by-domain silhouette={sil_domain:.3f}",
        fontsize=13, fontweight="bold")

    # Plot 1: colored by label
    label_colors = {0: "#2196F3", 1: "#F44336"}
    label_names  = {0: "normal", 1: "defective"}
    for lbl in [0, 1]:
        mask = label_arr == lbl
        axes[0].scatter(emb[mask, 0], emb[mask, 1],
                        c=label_colors[lbl], alpha=0.6, s=30,
                        label=label_names[lbl])
    axes[0].set_title("Colored by Label (good / defective)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Plot 2: colored by domain
    palette   = sns.color_palette("tab10", len(unique_domains))
    dom_colors = dict(zip(unique_domains, palette))
    for dom in unique_domains:
        mask = domain_arr == dom
        axes[1].scatter(emb[mask, 0], emb[mask, 1],
                        c=[dom_colors[dom]], alpha=0.6, s=30, label=dom)
    axes[1].set_title("Colored by Domain")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = RESULTS_DIR / f"{save_prefix}.png"
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()

    return sil_label, sil_domain, per_domain_sil


def main(args):
    device = get_device()
    print(f"Device: {device}")

    categories = [args.source] + args.targets
    # Deduplicate while preserving order
    seen = set()
    categories = [c for c in categories if not (c in seen or seen.add(c))]
    print(f"Sampling from: {categories}")

    all_paths, all_labels, all_domains = [], [], []
    for cat in categories:
        paths, labels, domains = sample_category(cat, N_PER_CLASS)
        all_paths  += paths
        all_labels += labels
        all_domains += domains
    print(f"Total: {len(all_paths)} images")

    results = {}

    # CNN features
    print("\n[CNN Features]")
    cnn_feats = extract_cnn_features(all_paths, args.source, device)
    sil_lbl, sil_dom, per_dom = compute_and_plot(
        cnn_feats, all_labels, all_domains,
        model_name=f"ResNet-50 (trained on {args.source})",
        save_prefix=f"exp6_tsne_cnn_{args.source}",
    )
    results["cnn"] = {"sil_label": sil_lbl, "sil_domain": sil_dom,
                      "per_domain": per_dom,
                      "mean_per_domain": float(np.nanmean(list(per_dom.values())))}

    # VLM features
    if args.no_adapter:
        adapter = None
    else:
        adapter = args.adapter or str(RESULTS_DIR / f"vlm_{args.source}_adapter")
    print("\n[VLM Features]")
    vlm_feats, vlm_label = extract_vlm_features(all_paths, device, adapter)
    sil_lbl, sil_dom, per_dom = compute_and_plot(
        vlm_feats, all_labels, all_domains,
        model_name=f"PaliGemma-3B ({vlm_label}, trained on {args.source})",
        save_prefix=f"exp6_tsne_vlm_{vlm_label}_{args.source}",
    )
    results[f"vlm_{vlm_label}"] = {
        "sil_label": sil_lbl, "sil_domain": sil_dom,
        "per_domain": per_dom,
        "mean_per_domain": float(np.nanmean(list(per_dom.values())))}

    # Summary
    print("\n" + "=" * 72)
    print("SILHOUETTE SCORE SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<35} {'Per-Domain':>12} {'By Domain':>12}")
    print("  " + "-" * 60)
    for mn, scores in results.items():
        print(f"  {mn:<35} {scores['mean_per_domain']:>12.4f} "
              f"{scores['sil_domain']:>12.4f}")
    print("=" * 72)
    print("Interpretation:")
    print("  High Per-Domain = good/defect well separated within each domain")
    print("  High By Domain  = features cluster by domain (domain-locked)")

    # Save to separate files to avoid overwriting
    for model_name, scores in results.items():
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        with open(RESULTS_DIR / f"exp6_silhouette_{safe_name}.json", "w") as f:
            json.dump(scores, f, indent=2)
    # Also save combined
    with open(RESULTS_DIR / "exp6_silhouette_scores.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="carpet", choices=TEXTURE_CATEGORIES)
    parser.add_argument("--targets", nargs="+", default=["leather", "tile", "wood"],
                        choices=TEXTURE_CATEGORIES)
    parser.add_argument("--adapter", default=None,
                        help="Path to VLM LoRA adapter")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Use zero-shot VLM features (no adapter)")
    main(parser.parse_args())
