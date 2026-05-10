"""
Dataset loader for MVTec AD texture categories.

MVTec AD folder structure:
  data/mvtec/<category>/train/good/     <- defect-free only (no defective!)
  data/mvtec/<category>/test/good/      <- defect-free test
  data/mvtec/<category>/test/<defect>/  <- defective test images

Since MVTec train split has ONLY good images, we create a supervised
train/val/test split by combining:
  - All train/good images           -> label 0 (normal)
  - 80% of test defective images    -> label 1 (defective, used for training)
  - 20% of test defective images    -> label 1 (defective, held out for test)
  - All test/good images            -> label 0 (held out for test)

This gives a clean supervised binary classification setup while
respecting that defective images only exist in the test folder.
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
BINARY_CLASSES     = ["good", "defective"]

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent.parent
MVTEC_DIR   = PROJECT_DIR / "data" / "mvtec"

# ── Image transform ───────────────────────────────────────────────────────────

def default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ── Sample loaders ────────────────────────────────────────────────────────────

def _collect_images(folder: Path) -> List[Path]:
    """Collect all images from a folder (non-recursive)."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
                  and p.is_file())


def _load_mvtec_samples(
    category: str,
    split: str,
    defect_train_ratio: float = 0.8,
    seed: int = 42,
) -> List[Tuple[Path, int]]:
    """
    Load (image_path, label) pairs for binary classification.

    split="train":
        - All train/good images (label=0)
        - 80% of test defective images (label=1)

    split="test":
        - All test/good images (label=0)
        - 20% of test defective images (label=1)
    """
    cat_dir = MVTEC_DIR / category
    if not cat_dir.exists():
        raise FileNotFoundError(
            f"MVTec category '{category}' not found at {cat_dir}\n"
            f"Download from https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )

    # Stratified split by defect type — ensures every defect type
    # is represented in both train and test sets
    test_dir     = cat_dir / "test"
    defect_train = []
    defect_test  = []
    rng          = random.Random(seed)

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name == "good":
            continue
        imgs    = sorted(_collect_images(cls_dir))
        rng.shuffle(imgs)
        n_train = max(1, int(defect_train_ratio * len(imgs)))
        defect_train.extend(imgs[:n_train])
        defect_test.extend(imgs[n_train:])

    samples = []

    if split == "train":
        # Good images from train/good
        for p in _collect_images(cat_dir / "train" / "good"):
            samples.append((p, 0))
        # Defective images (train portion)
        for p in defect_train:
            samples.append((p, 1))

    elif split == "test":
        # Good images from test/good
        for p in _collect_images(cat_dir / "test" / "good"):
            samples.append((p, 0))
        # Defective images (held-out portion)
        for p in defect_test:
            samples.append((p, 1))

    return samples


# ── Dataset ───────────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """Binary defect classification: good=0, defective=1."""

    def __init__(
        self,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert category in TEXTURE_CATEGORIES, \
            f"Unknown category: {category}. Choose from {TEXTURE_CATEGORIES}"
        assert split in ("train", "test")

        self.category = category
        self.split    = split
        self.classes  = BINARY_CLASSES
        self.samples  = _load_mvtec_samples(category, split)

        if not self.samples:
            raise RuntimeError(
                f"No images found for category={category} split={split}")

        self.transform = transform or default_transform()
        good   = sum(1 for _, l in self.samples if l == 0)
        defect = sum(1 for _, l in self.samples if l == 1)
        print(f"[{category} {split}] {len(self.samples)} images "
              f"(good={good}, defective={defect})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def get_loader(
    category: str,
    split: str,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    ds = MVTecDataset(category, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)
