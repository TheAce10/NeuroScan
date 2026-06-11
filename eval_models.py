"""Quick evaluation of all three trained models on the local test set."""
import os, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
WEIGHTS_ROOT = Path(__file__).parent / "weights" / "neuroscan"
TEST_DIR     = Path(__file__).parent / "weights" / "Testing"
DEVICE       = torch.device("cpu")
MEAN, STD    = [0.485,0.456,0.406], [0.229,0.224,0.225]

class TestSet(Dataset):
    def __init__(self, root, img_size):
        self.samples = []
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.tf = tf
        label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = root / cls
            if not d.is_dir(): continue
            for f in d.iterdir():
                if f.suffix.lower() in ('.jpg','.jpeg','.png'):
                    self.samples.append((f, label_map[cls]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, lbl = self.samples[i]
        return self.tf(Image.open(p).convert('RGB')), lbl

def eval_model(name, model, img_size):
    loader = DataLoader(TestSet(TEST_DIR, img_size), batch_size=64,
                        num_workers=0, shuffle=False)
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

configs = {
    'EfficientNetB3': {
        'weights': WEIGHTS_ROOT / 'efficientnet_b3' / 'outputs' / 'efficientnet_b3_neuroscan.pt',
        'img_size': 224,
        'build': lambda: _build_effnet(),
    },
    'VGG16': {
        'weights': WEIGHTS_ROOT / 'vgg16' / 'checkpoints' / 'phase2_best.pt',
        'img_size': 224,
        'build': lambda: _build_vgg16(),
    },
    'DenseNet121': {
        'weights': WEIGHTS_ROOT / 'densenet121' / 'outputs' / 'densenet121_neuroscan.pt',
        'img_size': 224,
        'build': lambda: _build_densenet121(),
    },
}

def _build_effnet():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 4))
    return m

def _build_vgg16():
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(*list(m.classifier.children())[:-1], nn.Dropout(0.4), nn.Linear(4096,4))
    return m

def _build_densenet121():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier.in_features, 4))
    return m

print(f'\nTest set: {TEST_DIR}')
print('='*60)
results = {}
for name, cfg in configs.items():
    if not cfg['weights'].exists():
        print(f'\n[SKIP] {name} — weights not found'); continue
    print(f'\nEvaluating {name}...')
    model = cfg['build']()
    model.load_state_dict(torch.load(cfg['weights'], map_location='cpu', weights_only=True))
    m = eval_model(name, model, cfg['img_size'])
    results[name] = m
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(m['report'])

print('\n' + '='*60)
print('MODEL COMPARISON SUMMARY')
print('='*60)
print(f"{'Model':<18} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print('-'*60)
for name, m in results.items():
    print(f"{name:<18} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
