"""
NeuroScan — Brain Tumor MRI Classification
Training script for EfficientNetB3, VGG16, DenseNet121, InceptionV3.

Two-phase transfer learning:
  Phase 1 — frozen backbone, train head only (AdamW, CosineAnnealingLR, 10 epochs)
  Phase 2 — unfreeze last conv blocks, fine-tune end-to-end (AdamW, CosineAnnealingLR, 15 epochs)

DenseNet121 uses FocalLoss + class weighting to address glioma/meningioma confusion.
InceptionV3 uses auxiliary classifier loss (weighted 0.4) during Phase 1 training.

Usage:
    python train.py --model efficientnet_b3 --data-dir ~/brain-tumor-mri --output-dir ~/neuroscan/efficientnet_b3
    python train.py --model vgg16           --data-dir ~/brain-tumor-mri --output-dir ~/neuroscan/vgg16
    python train.py --model densenet121     --data-dir ~/brain-tumor-mri --output-dir ~/neuroscan/densenet121
    python train.py --model inception_v3   --data-dir ~/brain-tumor-mri --output-dir ~/neuroscan/inception_v3
"""

import os, copy, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES   = 4
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]
# Upweight glioma (index 0) — lowest recall in baseline models
CLASS_WEIGHTS = torch.tensor([2.0, 1.2, 0.8, 0.8])


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='NeuroScan training')
    p.add_argument('--model',          required=True,
                   choices=['efficientnet_b3', 'vgg16', 'densenet121', 'inception_v3'])
    p.add_argument('--data-dir',       required=True,
                   help='Root of Brain Tumor MRI dataset (contains Training/ and Testing/)')
    p.add_argument('--output-dir',     required=True,
                   help='Where to save checkpoints and final weights')
    p.add_argument('--epochs-phase1',  type=int,   default=10)
    p.add_argument('--epochs-phase2',  type=int,   default=15)
    p.add_argument('--batch-size',     type=int,   default=32)
    p.add_argument('--lr-phase1',      type=float, default=1e-3)
    p.add_argument('--lr-phase2',      type=float, default=5e-5)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--num-workers',    type=int,   default=4)
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []
        self.label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.transform = transform
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(d, f), self.label_map[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_loaders(data_dir, img_size, batch_size, num_workers):
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
    pin = torch.cuda.is_available()
    train_set = BrainTumorDataset(os.path.join(data_dir, 'Training'), aug)
    test_set  = BrainTumorDataset(os.path.join(data_dir, 'Testing'),  basic)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    print(f'  Train: {len(train_set)}  |  Test: {len(test_set)}')
    return train_loader, test_loader


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss — reduces weight of easy examples to focus training on hard cases."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Model builders ────────────────────────────────────────────────────────────

def build_efficientnet_b3():
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
    )
    for p in m.features.parameters():
        p.requires_grad = False
    return m

def unfreeze_efficientnet_b3(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if any(k in name for k in ['features.6', 'features.7', 'features.8', 'classifier']):
            p.requires_grad = True


def build_vgg16():
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        *list(m.classifier.children())[:-1],   # keep first 6 layers, drop Linear(4096,1000)
        nn.Dropout(p=0.4),
        nn.Linear(4096, NUM_CLASSES),
    )
    for p in m.features.parameters():
        p.requires_grad = False
    return m

def unfreeze_vgg16(m):
    for p in m.parameters(): p.requires_grad = False
    for i, layer in enumerate(m.features):
        if i >= 24:    # block 5: Conv512 × 3 (indices 24-30)
            for p in layer.parameters(): p.requires_grad = True
    for p in m.classifier.parameters(): p.requires_grad = True


def build_densenet121():
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier.in_features, NUM_CLASSES),
    )
    for p in m.features.parameters():
        p.requires_grad = False
    return m

def unfreeze_densenet121(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if any(k in name for k in ['denseblock4', 'norm5', 'classifier']):
            p.requires_grad = True


def build_inception_v3():
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                             aux_logits=True)
    m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, NUM_CLASSES)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.fc.in_features, NUM_CLASSES),
    )
    for p in m.parameters(): p.requires_grad = False
    for p in m.fc.parameters():           p.requires_grad = True
    for p in m.AuxLogits.fc.parameters(): p.requires_grad = True
    return m

def unfreeze_inception_v3(m):
    for p in m.parameters(): p.requires_grad = False
    for name, p in m.named_parameters():
        if any(k in name for k in ['Mixed_7', 'fc', 'AuxLogits']):
            p.requires_grad = True


MODEL_REGISTRY = {
    'efficientnet_b3': {
        'build':     build_efficientnet_b3,
        'unfreeze':  unfreeze_efficientnet_b3,
        'img_size':  224,
        'use_focal': False,
        'inception': False,
        'output':    'efficientnet_b3_neuroscan.pt',
    },
    'vgg16': {
        'build':     build_vgg16,
        'unfreeze':  unfreeze_vgg16,
        'img_size':  224,
        'use_focal': False,
        'inception': False,
        'output':    'vgg16_neuroscan.pt',
    },
    'densenet121': {
        'build':     build_densenet121,
        'unfreeze':  unfreeze_densenet121,
        'img_size':  224,
        'use_focal': True,
        'inception': False,
        'output':    'densenet121_neuroscan.pt',
    },
    'inception_v3': {
        'build':     build_inception_v3,
        'unfreeze':  unfreeze_inception_v3,
        'img_size':  299,
        'use_focal': False,
        'inception': True,      # aux_logits during training
        'output':    'inception_v3_neuroscan.pt',
    },
}


# ── Training helpers ──────────────────────────────────────────────────────────

def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {t:,} / {n:,} ({100*t/n:.1f}%)')


def train_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, is_inception):
    model.train()
    loss_sum = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            if is_inception:
                out, aux = model(imgs)
                loss = criterion(out, labels) + 0.4 * criterion(aux, labels)
            else:
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
    for imgs, labels in tqdm(loader, leave=False, desc='  eval'):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)   # eval mode: InceptionV3 returns single tensor
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


def run_phase(model, train_loader, test_loader, criterion, epochs, lr,
              weight_decay, output_dir, device, phase_name, use_amp, is_inception):
    optimizer  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr, weight_decay=weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler     = torch.amp.GradScaler('cuda', enabled=use_amp)
    ckpt_dir   = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    history    = []

    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer,
                                      device, use_amp, scaler, is_inception)
        va_loss, va_acc = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': round(tr_loss, 6),
                        'train_acc': round(tr_acc, 6), 'val_loss': round(va_loss, 6),
                        'val_acc': round(va_acc, 6)})
        print(f'  [{phase_name}] {epoch:02d}/{epochs}  '
              f'train={tr_loss:.4f}/{tr_acc:.4f}  '
              f'val={va_loss:.4f}/{va_acc:.4f}  ({elapsed:.1f}s)')
        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(ckpt_dir, f'{phase_name}_best.pt'))
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'{phase_name}_epoch{epoch:02d}.pt'))

    model.load_state_dict(best_state)
    with open(os.path.join(ckpt_dir, f'{phase_name}_history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f'  Best val acc ({phase_name}): {best_acc:.4f}')
    return best_acc


@torch.no_grad()
def full_evaluation(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    y, p = np.array(labels), np.array(preds)
    metrics = {
        'accuracy':           round(float(accuracy_score(y, p)), 4),
        'f1_weighted':        round(float(f1_score(y, p, average='weighted')), 4),
        'precision_weighted': round(float(precision_score(y, p, average='weighted', zero_division=0)), 4),
        'recall_weighted':    round(float(recall_score(y, p, average='weighted')), 4),
    }
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1        : {metrics['f1_weighted']:.4f}")
    print(f"  Precision : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    : {metrics['recall_weighted']:.4f}\n")
    print(classification_report(y, p, target_names=CLASS_NAMES))
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    cfg     = MODEL_REGISTRY[args.model]

    print('=' * 60)
    print(f'  NeuroScan — {args.model}')
    print(f'  Device  : {device}' +
          (f'  ({torch.cuda.get_device_name(0)})' if use_amp else ''))
    print(f'  AMP     : {use_amp}')
    print(f'  Data    : {args.data_dir}')
    print(f'  Output  : {args.output_dir}')
    print('=' * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = build_loaders(
        args.data_dir, cfg['img_size'], args.batch_size, args.num_workers)

    model     = cfg['build']().to(device)
    criterion = (FocalLoss(gamma=2.0, weight=CLASS_WEIGHTS.to(device))
                 if cfg['use_focal'] else nn.CrossEntropyLoss())

    # Phase 1 — head only
    print('\n── Phase 1: frozen backbone ─────────────────────────────────')
    count_trainable(model)
    p1_acc = run_phase(model, train_loader, test_loader, criterion,
                       args.epochs_phase1, args.lr_phase1, args.weight_decay,
                       args.output_dir, device, 'phase1', use_amp, cfg['inception'])

    # Phase 2 — unfreeze last blocks
    print('\n── Phase 2: fine-tuning last blocks ─────────────────────────')
    cfg['unfreeze'](model)
    count_trainable(model)
    p2_acc = run_phase(model, train_loader, test_loader, criterion,
                       args.epochs_phase2, args.lr_phase2, args.weight_decay,
                       args.output_dir, device, 'phase2', use_amp, cfg['inception'])

    # Final evaluation
    print('\n── Final Evaluation (test set) ──────────────────────────────')
    metrics = full_evaluation(model, test_loader, device)

    # Save weights
    out_dir = os.path.join(args.output_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg['output'])
    torch.save(model.state_dict(), out_path)
    print(f'Weights saved: {out_path}')

    # Save summary
    summary = {
        'model': args.model, 'phase1_best_val_acc': p1_acc,
        'phase2_best_val_acc': p2_acc, 'final_test': metrics,
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Summary saved: {os.path.join(args.output_dir, "training_summary.json")}')


if __name__ == '__main__':
    main()
