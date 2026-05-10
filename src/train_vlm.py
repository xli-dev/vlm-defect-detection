"""
Experiment 5 (part 1): Fine-tune PaliGemma on source domain using LoRA.
Task: binary classification — normal vs defective.
After training, run linear_probe.py --backbone vlm to get cross-domain results.

Usage:
    python src/train_vlm.py --source carpet
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

from dataset import MVTecDataset, _load_mvtec_samples

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_ID = "google/paligemma-3b-mix-224"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

PROMPT = "<image> Does this surface have any visible defects such as scratches, holes, or cracks? Answer yes or no."


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"Source domain: {args.source}")

    # Load source domain training samples
    raw_samples = _load_mvtec_samples(args.source, split="train")
    # Include test defective images for training the VLM
    # (MVTec train split only has good images — we need defective too)
    test_samples = _load_mvtec_samples(args.source, split="test")
    defective = [(p, l) for p, l in test_samples if l == 1]
    # Use 80% of defective for training, 20% for validation
    rng = random.Random(42)
    rng.shuffle(defective)
    n_train = int(0.8 * len(defective))
    defective_train = defective[:n_train]

    label_map = {0: "no", 1: "yes"}
    all_samples = [(p, label_map[l]) for p, l in raw_samples + defective_train]
    print(f"Training on {len(all_samples)} images from {args.source}")

    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base      = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)
    model = get_peft_model(base, LORA_CONFIG).to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(all_samples)
        total_loss = 0.0

        for path, cls_name in tqdm(all_samples, desc=f"Epoch {epoch}"):
            img    = Image.open(path).convert("RGB")
            # Processor may add image tokens; use suffix for the answer
            inputs = processor(
                images=img,
                text=PROMPT,
                suffix=cls_name,   # answer the model should learn to generate
                return_tensors="pt",
            ).to(device)

            loss = model(**inputs).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Avg Loss: {total_loss / len(all_samples):.4f}")

    adapter_path = RESULTS_DIR / f"vlm_{args.source}_adapter"
    model.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to {adapter_path}")
    print(f"Next: python src/linear_probe.py --backbone vlm "
          f"--source {args.source} --target <target> "
          f"--adapter {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="carpet",
                        choices=["carpet", "grid", "leather", "tile", "wood"])
    parser.add_argument("--epochs",  type=int,   default=5)
    parser.add_argument("--lr",      type=float, default=2e-5)
    main(parser.parse_args())
