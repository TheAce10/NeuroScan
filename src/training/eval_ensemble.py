"""
NeuroScan — Ensemble evaluation.

Loads the three trained baseline models, runs the full test set through each,
soft-votes (averages softmax probabilities), and writes the resulting metrics
into results/model_comparison.json under the "ensemble" key so it appears
alongside the individual models in generate_figures.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)

ROOT         = Path(__file__).resolve().parent.parent.parent
WEIGHTS_ROOT = ROOT / "misc" / "weights"
TEST_DIR     = ROOT / "data" / "Testing"
RESULTS_PATH = ROOT / "results" / "model_comparison.json"
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
MEAN, STD    = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE     = 224
DEVICE       = torch.device("cpu")


class TestSet(Dataset):
    def __init__(self, tf):
        self.tf = tf
        label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
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


def build_efficientnet_b3():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.classifier[1].in_features, 4))
    return m


def build_vgg16():
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(*list(m.classifier.children())[:-1],
                                  nn.Dropout(p=0.4), nn.Linear(4096, 4))
    return m


def build_densenet121():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.classifier.in_features, 4))
    return m


MODELS = {
    "efficientnet_b3": {
        "builder": build_efficientnet_b3,
        "weights": WEIGHTS_ROOT / "efficientnet_b3" / "outputs" / "efficientnet_b3_neuroscan.pt",
    },
    "vgg16": {
        "builder": build_vgg16,
        "weights": WEIGHTS_ROOT / "vgg16" / "checkpoints" / "phase2_best.pt",
    },
    "densenet121": {
        "builder": build_densenet121,
        "weights": WEIGHTS_ROOT / "densenet121" / "outputs" / "densenet121_neuroscan.pt",
    },
}

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

loader = DataLoader(TestSet(tf), batch_size=32, num_workers=0, shuffle=False)

print(f"Test set: {TEST_DIR}  ({len(loader.dataset)} samples)")
print("=" * 60)

all_probs = []
y_true = None
for name, cfg in MODELS.items():
    if not cfg["weights"].exists():
        raise FileNotFoundError(f"Missing weights for {name}: {cfg['weights']}")
    print(f"Running {name} ...")
    model = cfg["builder"]()
    model.load_state_dict(torch.load(cfg["weights"], map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    probs_batches, labels_batches = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = F.softmax(model(imgs.to(DEVICE)), dim=1)
            probs_batches.append(p.cpu().numpy())
            labels_batches.append(lbls.numpy())
    all_probs.append(np.concatenate(probs_batches))
    if y_true is None:
        y_true = np.concatenate(labels_batches)
    del model

ensemble_probs = np.mean(all_probs, axis=0)
y_pred = ensemble_probs.argmax(axis=1)

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
per_class = {
    cls: {
        "precision": round(report[cls]["precision"], 4),
        "recall":    round(report[cls]["recall"], 4),
        "f1":        round(report[cls]["f1-score"], 4),
        "support":   int(report[cls]["support"]),
    }
    for cls in CLASS_NAMES
}
metrics = {
    "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
    "f1_weighted":         round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    "precision_weighted":  round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    "recall_weighted":     round(float(recall_score(y_true, y_pred, average="weighted")), 4),
    "per_class": per_class,
}

print("\nEnsemble (soft vote) results")
print("-" * 60)
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"F1        : {metrics['f1_weighted']:.4f}")
print(f"Precision : {metrics['precision_weighted']:.4f}")
print(f"Recall    : {metrics['recall_weighted']:.4f}")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

with open(RESULTS_PATH) as fh:
    results = json.load(fh)
results["ensemble"] = metrics
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved ensemble metrics into {RESULTS_PATH}")
