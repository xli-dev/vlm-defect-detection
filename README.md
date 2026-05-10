# Cross-Domain Industrial Defect Detection with Vision-Language Models

## Motivation

In industrial quality control, defect detection models are typically trained on images from one production environment and then deployed in another — different equipment, lighting conditions, or materials. This **domain shift** causes significant performance degradation even when the underlying defect types are the same, making real-world deployment unreliable.

This project investigates whether Vision-Language Models (VLMs), with their semantically rich pre-trained representations, generalize better across visual domains than traditional CNNs — using the MVTec AD texture dataset as a controlled benchmark for cross-domain industrial defect detection.

## Research Question

> When trained to detect defects on one texture domain (carpet) and evaluated on visually different texture domains (leather, tile, wood), do VLM features transfer significantly better than CNN features?

## Dataset

**MVTec AD** — 5 texture categories (carpet, grid, leather, tile, wood), each with normal and defective images. Source domain: carpet. Target domains: leather, tile, wood.

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

```
data/mvtec/
  carpet/train/good/    carpet/test/good/    carpet/test/<defect_type>/
  leather/              tile/                wood/
```

## Experiments & Results

### Phase 1: In-Domain Performance (trained and tested on carpet)

| Model | Protocol | Carpet Test AUC |
|---|---|---|
| ResNet-50 CNN | Direct inference | 0.9482 |
| PaliGemma pre-trained | Full linear probe | 1.0000 |
| PaliGemma fine-tuned on carpet | Full linear probe | 1.0000 |

### Phase 2: Cross-Domain Transfer (5-shot linear probe, carpet → target)

| Model | leather | tile | wood |
|---|---|---|---|
| ResNet-50 CNN | 0.3781 | 0.6459 | 0.1654 |
| PaliGemma pre-trained | **1.0000** | **1.0000** | **1.0000** |
| PaliGemma fine-tuned on carpet | **1.0000** | **1.0000** | **1.0000** |

### Phase 3: VLM Zero-Shot Generative Detection (no training data at all)

| Domain | AUC | Accuracy |
|---|---|---|
| carpet | 0.5750 | 64.6% |
| leather | 0.6000 | 69.2% |
| tile | 0.6053 | 71.2% |
| wood | 0.9286 | 93.9% |

### Phase 4: Per-Domain Silhouette Score (good/defect separation within each domain)

| Domain | CNN | PaliGemma pre-trained | PaliGemma fine-tuned |
|---|---|---|---|
| carpet | 0.4515 | 0.3497 | 0.4281 |
| leather | 0.2441 | 0.4095 | 0.4293 |
| tile | 0.2187 | 0.3008 | 0.3085 |
| wood | 0.1107 | 0.2169 | 0.2281 |
| **Mean** | **0.2562** | **0.3192** | **0.3485** |

## Key Findings

1. **CNN features fail severely under domain shift** — AUC drops from 0.95 (in-domain) to 0.17–0.65 on target domains, with near-zero within-domain silhouette on wood (0.11).

2. **VLM pre-training alone produces perfectly transferable features** — PaliGemma pre-trained features achieve AUC 1.0 on all target domains with just 5 labeled samples, without any task-specific fine-tuning.

3. **Fine-tuning on source domain adds marginal benefit** — fine-tuned features slightly improve per-domain silhouette (0.35 vs 0.32) but linear probe AUC is already saturated at 1.0 for the pre-trained model.

4. **VLM zero-shot generative detection is domain-dependent** — works well on wood (AUC 0.93) but struggles on carpet (AUC 0.58), suggesting visual complexity affects prompt-based detection.

5. **The key mechanism** — VLM features cluster more by domain (by-domain silhouette 0.61 vs 0.44 for CNN) but maintain significantly better good/defective separability within each domain cluster (mean per-domain silhouette 0.32–0.35 vs 0.26 for CNN).

## Setup

```bash
git clone https://github.com/yourusername/vlm-defect-detection
cd vlm-defect-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Phase 1: Train models on carpet
python src/train_cnn.py --source carpet
python src/train_vlm.py --source carpet

# Phase 1: In-domain evaluation
python src/linear_probe.py --backbone vlm --source carpet --target carpet
python src/linear_probe.py --backbone vlm --source carpet --target carpet --adapter results/vlm_carpet_adapter

# Phase 2: Cross-domain 5-shot linear probe
python src/linear_probe.py --backbone cnn --source carpet --target leather --n_shot 5
python src/linear_probe.py --backbone cnn --source carpet --target tile    --n_shot 5
python src/linear_probe.py --backbone cnn --source carpet --target wood    --n_shot 5

python src/linear_probe.py --backbone vlm --source carpet --target leather --n_shot 5
python src/linear_probe.py --backbone vlm --source carpet --target tile    --n_shot 5
python src/linear_probe.py --backbone vlm --source carpet --target wood    --n_shot 5

python src/linear_probe.py --backbone vlm --source carpet --target leather --adapter results/vlm_carpet_adapter --n_shot 5
python src/linear_probe.py --backbone vlm --source carpet --target tile    --adapter results/vlm_carpet_adapter --n_shot 5
python src/linear_probe.py --backbone vlm --source carpet --target wood    --adapter results/vlm_carpet_adapter --n_shot 5

# Phase 3: VLM zero-shot generative detection
python src/eval_vlm_zeroshot.py --category carpet
python src/eval_vlm_zeroshot.py --category leather
python src/eval_vlm_zeroshot.py --category tile
python src/eval_vlm_zeroshot.py --category wood

# Phase 4: Feature visualization + silhouette scores
python src/visualize_features.py --source carpet --targets carpet leather tile wood
python src/visualize_features.py --source carpet --targets carpet leather tile wood --no_adapter

# Summary
python src/summarize_results.py
```

## Project Structure

```
vlm-defect-detection/
├── data/mvtec/                   # MVTec AD texture categories
├── src/
│   ├── dataset.py                # MVTec dataset loader
│   ├── train_cnn.py              # CNN baseline training
│   ├── linear_probe.py           # Cross-domain linear probe evaluation
│   ├── eval_vlm_zeroshot.py      # VLM zero-shot generative detection
│   ├── train_vlm.py              # PaliGemma LoRA fine-tuning
│   ├── visualize_features.py     # t-SNE + silhouette score analysis
│   └── summarize_results.py      # Print full results table
├── results/                      # Saved weights, metrics, plots
├── requirements.txt
└── README.md
```

## Tech Stack

- PyTorch (MPS backend for Apple Silicon)
- Hugging Face Transformers + PEFT (LoRA fine-tuning)
- PaliGemma-3B-mix (Vision-Language Model)
- scikit-learn (linear probe, silhouette score, t-SNE)
- matplotlib / seaborn (visualization)
