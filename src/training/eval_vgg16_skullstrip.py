"""
Compare baseline VGG16 vs skull-strip VGG16 on the local test set.

The skull-strip model was trained with SkullStripTransform applied before
normalisation, so the same preprocessing is applied here at eval time.
The baseline model uses plain resize + normalise.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from scipy.ndimage import binary_fill_holes

ROOT         = Path(__file__).resolve().parent.parent.parent
WEIGHTS_ROOT = ROOT / "misc" / "weights"
TEST_DIR     = ROOT / "data" / "Testing"
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
MEAN, STD    = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE     = 224


# ---------------------------------------------------------------------------
# SkullStripTransform (must match the one used during training)
# ---------------------------------------------------------------------------
class SkullStripTransform:
    def __init__(self, closing_kernel=25, center_sigma=0.33, edge_floor=0.15):
        self.close_k      = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        self.center_sigma = center_sigma
        self.edge_floor   = edge_floor

    def _center_bias(self, gray):
        h, w = gray.shape
        Y, X = np.ogrid[:h, :w]
        sigma  = min(h, w) * self.center_sigma
        weight = np.exp(-((X - w // 2) ** 2 + (Y - h // 2) ** 2) / (2 * sigma ** 2))
        weight = self.edge_floor + (1.0 - self.edge_floor) * weight
        return (gray.astype(np.float32) * weight).clip(0, 255).astype(np.uint8)

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        biased = self._center_bias(gray)
        _, mask = cv2.threshold(biased, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_k)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask    = (labels == largest).astype(np.uint8) * 255
        mask = binary_fill_holes(mask > 0).astype(np.uint8) * 255
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
        return Image.fromarray((img_np.astype(np.float32) * mask_3ch).astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestSet(Dataset):
    def __init__(self, tf):
        self.tf      = tf
        label_map    = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples = []
        for cls in CLASS_NAMES:
            d = TEST_DIR / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((f, label_map[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, lbl = self.samples[i]
        return self.tf(Image.open(p).convert('RGB')), lbl


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_vgg16():
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(
        *list(m.classifier.children())[:-1],
        nn.Dropout(p=0.4),
        nn.Linear(4096, len(CLASS_NAMES)),
    )
    return m


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs).argmax(1).numpy())
            labels.extend(lbls.numpy())
    y, p = np.array(labels), np.array(preds)
    return {
        'accuracy':  round(accuracy_score(y, p), 4),
        'f1':        round(f1_score(y, p, average='weighted'), 4),
        'precision': round(precision_score(y, p, average='weighted', zero_division=0), 4),
        'recall':    round(recall_score(y, p, average='weighted'), 4),
        'report':    classification_report(y, p, target_names=CLASS_NAMES),
    }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
skull_strip = SkullStripTransform()

tf_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tf_stripped = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    skull_strip,
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
configs = {
    'VGG16 Baseline': {
        'weights': WEIGHTS_ROOT / 'vgg16' / 'checkpoints' / 'phase2_best.pt',
        'tf':      tf_baseline,
    },
    'VGG16 Skull-Strip': {
        'weights': WEIGHTS_ROOT / 'vgg16_skullstrip_neuroscan.pt',
        'tf':      tf_stripped,
    },
}

print(f'\nTest set : {TEST_DIR}')
print(f'Samples  : {len(list(TEST_DIR.glob("*/*.jpg")) + list(TEST_DIR.glob("*/*.png")))}')
print('=' * 65)

results = {}
for name, cfg in configs.items():
    w = cfg['weights']
    if not w.exists():
        print(f'\n[SKIP] {name} — weights not found at {w}')
        continue
    print(f'\nEvaluating {name} ...')
    model = build_vgg16()
    model.load_state_dict(torch.load(w, map_location='cpu', weights_only=True))
    loader = DataLoader(TestSet(cfg['tf']), batch_size=32,
                        num_workers=0, shuffle=False)
    m = evaluate(model, loader)
    results[name] = m
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print()
    print(m['report'])

print('=' * 65)
print('COMPARISON SUMMARY')
print('=' * 65)
print(f"{'Model':<24} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print('-' * 65)
for name, m in results.items():
    print(f"{name:<24} {m['accuracy']:>10.4f} {m['f1']:>8.4f} "
          f"{m['precision']:>10.4f} {m['recall']:>8.4f}")

if len(results) == 2:
    names = list(results)
    base, strip = results[names[0]], results[names[1]]
    delta_acc = strip['accuracy'] - base['accuracy']
    delta_f1  = strip['f1']       - base['f1']
    sign = lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
    print('-' * 65)
    print(f"{'Delta (strip - base)':<24} {sign(delta_acc):>10} {sign(delta_f1):>8}")
