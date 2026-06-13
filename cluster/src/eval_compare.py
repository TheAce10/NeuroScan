"""
NeuroScan — Model Evaluation & Comparison
Loads all trained models, evaluates on the full test set, outputs metrics (JSON + CSV),
confusion matrices, accuracy/loss curves, and per-class comparison bar charts.

Usage:
    python eval_compare.py \\
        --data-dir   ~/brain-tumor-mri \\
        --weights-dir ~/neuroscan \\
        --output-dir  ~/neuroscan/results

Weight file names expected (one file per model in weights-dir, searched recursively):
    efficientnet_b3_neuroscan.pt
    vgg16_neuroscan.pt
    densenet121_neuroscan.pt
    inception_v3_neuroscan.pt
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report,
                              confusion_matrix)
import matplotlib
matplotlib.use('Agg')          # no display needed on cluster
import matplotlib.pyplot as plt
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

WEIGHT_FILES = {
    'efficientnet_b3': 'efficientnet_b3_neuroscan.pt',
    'vgg16':           'vgg16_neuroscan.pt',
    'densenet121':     'densenet121_neuroscan.pt',
    'inception_v3':    'inception_v3_neuroscan.pt',
}


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='NeuroScan evaluation')
    p.add_argument('--data-dir',    required=True,
                   help='Root of Brain Tumor MRI dataset (contains Testing/)')
    p.add_argument('--weights-dir', required=True,
                   help='Root directory to search for .pt weight files')
    p.add_argument('--output-dir',  required=True,
                   help='Where to save metrics files and figures')
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    def __init__(self, root, transform):
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
        return self.transform(Image.open(path).convert('RGB')), label


def build_loader(test_dir, img_size, batch_size, num_workers):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = BrainTumorDataset(test_dir, tfm)
    pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin)


# ── Model builders (must match training exactly) ───────────────────────────────

def build_efficientnet_b3():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
    )
    return m

def build_vgg16():
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(
        *list(m.classifier.children())[:-1],
        nn.Dropout(p=0.4),
        nn.Linear(4096, NUM_CLASSES),
    )
    return m

def build_densenet121():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier.in_features, NUM_CLASSES),
    )
    return m

def build_inception_v3():
    m = models.inception_v3(weights=None, aux_logits=True)
    m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, NUM_CLASSES)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.fc.in_features, NUM_CLASSES),
    )
    return m


MODEL_REGISTRY = {
    'efficientnet_b3': {'builder': build_efficientnet_b3, 'img_size': 224},
    'vgg16':           {'builder': build_vgg16,           'img_size': 224},
    'densenet121':     {'builder': build_densenet121,     'img_size': 224},
    'inception_v3':    {'builder': build_inception_v3,    'img_size': 299},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_weight(search_root, filename):
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        out = model(imgs.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(labels), np.array(preds)


def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    per_class = {
        cls: {
            'precision': round(report[cls]['precision'], 4),
            'recall':    round(report[cls]['recall'],    4),
            'f1':        round(report[cls]['f1-score'],  4),
            'support':   int(report[cls]['support']),
        }
        for cls in CLASS_NAMES
    }
    return {
        'accuracy':            round(float(accuracy_score(y_true, y_pred)), 4),
        'f1_weighted':         round(float(f1_score(y_true, y_pred, average='weighted')), 4),
        'precision_weighted':  round(float(precision_score(y_true, y_pred,
                                                            average='weighted', zero_division=0)), 4),
        'recall_weighted':     round(float(recall_score(y_true, y_pred, average='weighted')), 4),
        'per_class':           per_class,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_id, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_id} — Confusion Matrix')
    plt.tight_layout()
    path = os.path.join(out_dir, f'confusion_{model_id}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_accuracy_comparison(all_results, out_dir):
    models_sorted = sorted(all_results, key=lambda m: all_results[m]['accuracy'], reverse=True)
    accs = [all_results[m]['accuracy'] for m in models_sorted]
    colors = ['#2ecc71' if a == max(accs) else '#3498db' for a in accs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models_sorted, accs, color=colors)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('NeuroScan — Model Accuracy Comparison')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_per_class_recall(all_results, out_dir):
    model_ids = list(all_results.keys())
    x = np.arange(NUM_CLASSES)
    width = 0.8 / max(len(model_ids), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mid in enumerate(model_ids):
        recalls = [all_results[mid]['per_class'][c]['recall'] for c in CLASS_NAMES]
        ax.bar(x + i * width - (len(model_ids) - 1) * width / 2,
               recalls, width=width * 0.9, label=mid)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Recall')
    ax.set_title('NeuroScan — Per-class Recall Comparison')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'per_class_recall.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_f1_comparison(all_results, out_dir):
    model_ids = list(all_results.keys())
    x = np.arange(NUM_CLASSES)
    width = 0.8 / max(len(model_ids), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mid in enumerate(model_ids):
        f1s = [all_results[mid]['per_class'][c]['f1'] for c in CLASS_NAMES]
        ax.bar(x + i * width - (len(model_ids) - 1) * width / 2,
               f1s, width=width * 0.9, label=mid)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_title('NeuroScan — Per-class F1 Comparison')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'per_class_f1.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_training_curves(output_base_dir, model_id, out_dir):
    """Read phase1/phase2 history JSONs from training output and plot combined curves."""
    model_out = os.path.join(output_base_dir, model_id)
    ckpt_dir  = os.path.join(model_out, 'checkpoints')

    combined_train_acc = []
    combined_val_acc   = []
    for phase in ['phase1', 'phase2']:
        hist_path = os.path.join(ckpt_dir, f'{phase}_history.json')
        if not os.path.exists(hist_path):
            return
        with open(hist_path) as fh:
            hist = json.load(fh)
        combined_train_acc.extend([h['train_acc'] for h in hist])
        combined_val_acc.extend([h['val_acc']   for h in hist])

    p1_len = len(combined_train_acc) // 2   # approximate; may differ if epochs differ
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(combined_train_acc) + 1)
    ax.plot(epochs, combined_train_acc, label='Train Acc')
    ax.plot(epochs, combined_val_acc,   label='Val Acc')
    ax.axvline(x=p1_len + 0.5, color='gray', linestyle='--', label='P1→P2')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_id} — Training Curve')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f'training_curve_{model_id}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('  NeuroScan — Model Evaluation')
    print(f'  Device    : {device}')
    print(f'  Data      : {args.data_dir}')
    print(f'  Weights   : {args.weights_dir}')
    print(f'  Output    : {args.output_dir}')
    print('=' * 60)

    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(figures_dir,     exist_ok=True)

    test_dir = os.path.join(args.data_dir, 'Testing')
    all_results = {}
    all_preds   = {}

    for model_id, cfg in MODEL_REGISTRY.items():
        wname = WEIGHT_FILES[model_id]
        wpath = find_weight(args.weights_dir, wname)
        if wpath is None:
            print(f'\n[SKIP] {model_id} — {wname} not found under {args.weights_dir}')
            continue

        print(f'\n{"="*60}')
        print(f'  Model   : {model_id}')
        print(f'  Weights : {wpath}')

        loader = build_loader(test_dir, cfg['img_size'], args.batch_size, args.num_workers)
        m = cfg['builder']()
        m.load_state_dict(torch.load(wpath, map_location=device))
        m = m.to(device)

        y_true, y_pred = evaluate(m, loader, device)
        metrics        = compute_metrics(y_true, y_pred)
        all_results[model_id] = metrics
        all_preds[model_id]   = (y_true, y_pred)

        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1        : {metrics['f1_weighted']:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        plot_confusion_matrix(y_true, y_pred, model_id, figures_dir)
        plot_training_curves(args.weights_dir, model_id, figures_dir)

        del m
        if device.type == 'cuda': torch.cuda.empty_cache()

    if not all_results:
        print('\nNo models evaluated — check --weights-dir path and weight file names.')
        return

    # Comparison plots
    print('\n── Generating comparison plots ──────────────────────────────')
    plot_accuracy_comparison(all_results, figures_dir)
    plot_per_class_recall(all_results, figures_dir)
    plot_f1_comparison(all_results, figures_dir)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'model_comparison.json')
    with open(json_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f'\nJSON saved : {json_path}')

    # Save flat CSV
    rows = []
    for model_id, m in all_results.items():
        row = {'model': model_id, 'accuracy': m['accuracy'],
               'f1_weighted': m['f1_weighted'],
               'precision_weighted': m['precision_weighted'],
               'recall_weighted': m['recall_weighted']}
        for cls in CLASS_NAMES:
            row[f'{cls}_precision'] = m['per_class'][cls]['precision']
            row[f'{cls}_recall']    = m['per_class'][cls]['recall']
            row[f'{cls}_f1']        = m['per_class'][cls]['f1']
        rows.append(row)
    df = pd.DataFrame(rows).set_index('model')
    csv_path = os.path.join(args.output_dir, 'model_comparison.csv')
    df.to_csv(csv_path)
    print(f'CSV saved  : {csv_path}')

    # Summary table
    print('\n' + '=' * 72)
    print('NEUROSCAN — MODEL COMPARISON SUMMARY')
    print('Test set: 400 samples × 4 classes = 1600 total')
    print('=' * 72)
    cols = ['accuracy', 'f1_weighted', 'recall_weighted',
            'glioma_recall', 'meningioma_recall', 'notumor_recall', 'pituitary_recall']
    summary = df[cols].sort_values('accuracy', ascending=False)
    summary.columns = ['Accuracy', 'F1', 'Recall',
                        'Glioma-R', 'Menin-R', 'NoTumor-R', 'Pituit-R']
    print(summary.to_string(float_format='{:.4f}'.format))
    best = summary.index[0]
    print(f'\n  Best overall : {best}  ({summary.loc[best, "Accuracy"]:.4f} accuracy)')
    print(f'  Best recall  : {summary["Recall"].idxmax()}  ({summary["Recall"].max():.4f})')
    print(f'  Best glioma  : {summary["Glioma-R"].idxmax()}  ({summary["Glioma-R"].max():.4f})')
    print(f'  Best menin   : {summary["Menin-R"].idxmax()}  ({summary["Menin-R"].max():.4f})')


if __name__ == '__main__':
    main()
