"""
NeuroScan — Multi-Architecture Brain Tumor MRI Classification
Training script for EfficientNetB3, VGG16, and DenseNet121.

Runs on CPU or GPU automatically — no code changes needed.
Phase 1: frozen backbone, train classification head only.
Phase 2: unfreeze last convolutional blocks, fine-tune end-to-end at lower learning rate.
DenseNet121 uses focal loss + class weighting to address glioma/meningioma confusion.

Usage:
    python train.py --model efficientnet_b3 \
                    --data-dir /shared/home/elikem/group2/brain-tumor-mri \
                    --output-dir /shared/home/elikem/group2/neuroscan/efficientnet_b3 \
                    --epochs-phase1 10 --epochs-phase2 10

    python train.py --model vgg16 ...
    python train.py --model densenet121 ...
"""

import os
import copy
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
)
from tqdm import tqdm


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NeuroScan Multi-Architecture Training")
    parser.add_argument("--model",         type=str, default="efficientnet_b3",
                        choices=["efficientnet_b3", "vgg16", "densenet121"],
                        help="Model architecture to train")
    parser.add_argument("--data-dir",      type=str,
                        default="/shared/home/elikem/group2/brain-tumor-mri")
    parser.add_argument("--output-dir",    type=str,
                        default="/shared/home/elikem/group2/neuroscan")
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--lr-phase1",     type=float, default=1e-3)
    parser.add_argument("--lr-phase2",     type=float, default=5e-5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--num-workers",   type=int, default=2)
    return parser.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Upweight glioma (lowest recall in baseline models)
CLASS_WEIGHTS = torch.tensor([2.0, 1.2, 0.8, 0.8])


class BrainTumorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples  = []
        self.label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.transform = transform
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(d, f), self.label_map[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_loaders(data_dir, batch_size, num_workers, img_size=224):
    aug = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    basic = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = BrainTumorDataset(os.path.join(data_dir, "Training"), aug)
    test_set  = BrainTumorDataset(os.path.join(data_dir, "Testing"),  basic)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    print(f"Train: {len(train_set)}  |  Test: {len(test_set)}")
    return train_loader, test_loader


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples, focuses on hard boundary cases."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Models ────────────────────────────────────────────────────────────────────

def build_efficientnet_b3(freeze_backbone=True):
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    if freeze_backbone:
        for p in m.features.parameters(): p.requires_grad = False
    return m

def unfreeze_efficientnet_b3(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if any(k in name for k in ["features.6", "features.7", "classifier"]):
            p.requires_grad = True


def build_vgg16(freeze_backbone=True):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(*list(m.classifier.children())[:-1],
                                  nn.Dropout(p=0.4), nn.Linear(4096, NUM_CLASSES))
    if freeze_backbone:
        for p in m.features.parameters(): p.requires_grad = False
    return m

def unfreeze_vgg16(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if "classifier" in name: p.requires_grad = True; continue
        if name.startswith("features.") and int(name.split(".")[1]) >= 24:
            p.requires_grad = True


def build_densenet121(freeze_backbone=True):
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.classifier.in_features, NUM_CLASSES))
    if freeze_backbone:
        for p in m.features.parameters(): p.requires_grad = False
    return m

def unfreeze_densenet121(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if any(k in name for k in ["denseblock4", "norm5", "classifier"]):
            p.requires_grad = True


def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {t:,} / {n:,} ({100*t/n:.1f}%)")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    loss_sum = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


def run_phase(model, train_loader, test_loader, criterion, optimizer, scheduler,
              epochs, device, phase_name, output_dir, use_amp):
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        va_loss, va_acc = eval_epoch(model, test_loader, criterion, device)
        if scheduler: scheduler.step()
        print(f"  [{phase_name}] Epoch {epoch:02d}/{epochs}  "
              f"train={tr_loss:.4f}/{tr_acc:.4f}  val={va_loss:.4f}/{va_acc:.4f}  "
              f"({time.time()-t0:.1f}s)")
        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(ckpt_dir, f"{phase_name}_best.pt"))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{phase_name}_epoch{epoch:02d}.pt"))

    model.load_state_dict(best_state)
    print(f"  Best val acc ({phase_name}): {best_acc:.4f}")


# ── Full evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    y, p = np.array(labels), np.array(preds)
    print(f"Accuracy  : {accuracy_score(y, p):.4f}")
    print(f"F1        : {f1_score(y, p, average='weighted'):.4f}")
    print(f"Precision : {precision_score(y, p, average='weighted', zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y, p, average='weighted'):.4f}")
    print()
    print(classification_report(y, p, target_names=CLASS_NAMES))


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "efficientnet_b3": {
        "build":    build_efficientnet_b3,
        "unfreeze": unfreeze_efficientnet_b3,
        "img_size": 224,
        "output":   "efficientnet_b3_neuroscan.pt",
        "use_focal": False,
    },
    "vgg16": {
        "build":    build_vgg16,
        "unfreeze": unfreeze_vgg16,
        "img_size": 224,
        "output":   "vgg16_neuroscan.pt",
        "use_focal": False,
    },
    "densenet121": {
        "build":    build_densenet121,
        "unfreeze": unfreeze_densenet121,
        "img_size": 224,
        "output":   "densenet121_neuroscan.pt",
        "use_focal": True,
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    cfg     = MODEL_REGISTRY[args.model]

    print(f"\nModel  : {args.model}")
    print(f"Device : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if use_amp else ""))
    print(f"AMP    : {use_amp}\n")

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, cfg["img_size"]
    )

    model = cfg["build"](freeze_backbone=True).to(device)

    if cfg["use_focal"]:
        criterion = FocalLoss(gamma=2.0, weight=CLASS_WEIGHTS.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1 — classifier only
    if args.epochs_phase1 > 0:
        print("\n── Phase 1: frozen backbone ─────────────────────────────")
        count_trainable(model)
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr_phase1, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_phase1)
        run_phase(model, train_loader, test_loader, criterion, opt, sch,
                  args.epochs_phase1, device, "phase1", args.output_dir, use_amp)

    # Phase 2 — unfreeze last blocks
    if args.epochs_phase2 > 0:
        print("\n── Phase 2: fine-tuning last blocks ─────────────────────")
        cfg["unfreeze"](model)
        count_trainable(model)
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr_phase2, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_phase2)
        run_phase(model, train_loader, test_loader, criterion, opt, sch,
                  args.epochs_phase2, device, "phase2", args.output_dir, use_amp)

    print("\n── Final Evaluation ─────────────────────────────────────")
    full_evaluation(model, test_loader, device)

    out_path = os.path.join(args.output_dir, "outputs", cfg["output"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\nWeights saved: {out_path}\n")


if __name__ == "__main__":
    main()
