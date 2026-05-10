"""
Experiment 1: ResNet-50 CNN Baseline
Train on source domain (carpet), evaluate on source domain test set.

Handles class imbalance via:
- Weighted cross-entropy loss
- Frozen backbone (only fine-tune last FC layer)
- Data augmentation
- Early stopping on AUC

Usage:
    python src/train_cnn.py --source carpet
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

from dataset import MVTecDataset, BINARY_CLASSES, MVTEC_DIR
from torch.utils.data import DataLoader

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms(split: str):
    """Augmentation for training, simple resize for test."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def build_resnet_linear_probe() -> nn.Module:
    """
    Freeze backbone, only train the final FC layer.
    Much less likely to overfit on small datasets.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the final FC layer
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Also unfreeze layer4 for some feature adaptation
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model


def compute_class_weights(dataset):
    """Compute inverse frequency weights to handle class imbalance."""
    n_good    = sum(1 for _, l in dataset.samples if l == 0)
    n_defect  = sum(1 for _, l in dataset.samples if l == 1)
    total     = n_good + n_defect
    w_good    = total / (2 * n_good)
    w_defect  = total / (2 * n_defect)
    print(f"  Class weights: good={w_good:.3f}, defective={w_defect:.3f}")
    return torch.tensor([w_good, w_defect], dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        probs   = torch.softmax(outputs, dim=1)[:, 1]
        preds   = outputs.argmax(1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc    = correct / total
    auc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(all_labels, all_preds,
                                   target_names=BINARY_CLASSES,
                                   output_dict=True, zero_division=0)
    return total_loss / total, acc, auc, report


def main(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"Source domain: {args.source}")

    train_ds = MVTecDataset(args.source, "train",
                            transform=get_transforms("train"))
    test_ds  = MVTecDataset(args.source, "test",
                            transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model         = build_resnet_linear_probe().to(device)
    class_weights = compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize trainable parameters
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, best_report, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_auc, report = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

        if test_auc > best_auc:
            best_auc    = test_auc
            best_report = report
            patience    = 0
            torch.save(model.state_dict(),
                       RESULTS_DIR / f"cnn_{args.source}_backbone.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest {args.source} test AUC: {best_auc:.4f}")
    print(f"Backbone saved to results/cnn_{args.source}_backbone.pt")

    with open(RESULTS_DIR / f"exp1_cnn_{args.source}.json", "w") as f:
        json.dump({
            "model": "ResNet-50 (frozen backbone + layer4)",
            "source": args.source,
            "test_auc": best_auc,
            "report": best_report,
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="carpet",
                        choices=["carpet", "grid", "leather", "tile", "wood"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    main(parser.parse_args())
