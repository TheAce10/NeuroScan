#!/usr/bin/env python3
"""
generate_figures.py — NeuroScan publication figures
CCRE standards: >=300 dpi, PDF+PNG output, colour-blind-safe palette
(Wong 2011), panel labels (a)-(f), 95% CI error bars, three-line table style.

Figures produced
----------------
fig1_model_overview.{pdf,png}   — overall accuracy / F1 / precision / recall
fig2_recall_heatmap.{pdf,png}   — per-class recall heatmap (3 models x 4 classes)
fig3_per_class_metrics.{pdf,png}— per-class P / R / F1 grouped bars (3 panels)
fig4_training_curves.{pdf,png}  — loss + accuracy curves, Phase 1 & 2 (3 models)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "model_comparison.json"
WEIGHTS = ROOT / "weights"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
with open(RESULTS) as fh:
    results = json.load(fh)

MODELS       = ["efficientnet_b3", "vgg16", "densenet121"]
MODEL_LABELS = ["EfficientNetB3",  "VGG16", "DenseNet121"]
CLASSES      = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumour", "Pituitary"]
N_TOTAL      = 1600   # total test samples
N_CLASS      = 400    # samples per class

# ── Wong (2011) colour-blind-safe palette ──────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"

MODEL_COLORS  = [BLUE, ORANGE, GREEN]
METRIC_COLORS = [BLUE, ORANGE, GREEN]   # Precision / Recall / F1

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Helpers ────────────────────────────────────────────────────────────────────
def ci95(p, n):
    """95% CI for a proportion (normal approximation)."""
    p = np.clip(p, 0, 1)
    return 1.96 * np.sqrt(p * (1 - p) / n)


def panel_label(ax, letter, x=-0.13, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", format=ext)
        print(f"  -> {stem}.{ext}")
    plt.close(fig)


def load_history(model_key):
    """Load and concatenate phase1 + phase2 history JSONs."""
    base = WEIGHTS / model_key / "checkpoints"
    with open(base / "phase1_history.json") as fh:
        p1 = json.load(fh)
    with open(base / "phase2_history.json") as fh:
        p2 = json.load(fh)
    n1 = len(p1)
    combined = list(p1)
    for row in p2:
        combined.append({k: (row[k] + n1 if k == "epoch" else row[k])
                         for k in row})
    return combined, n1


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overall model performance
# ══════════════════════════════════════════════════════════════════════════════
def fig1_model_overview():
    metric_keys    = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    metric_labels  = ["Accuracy", "Weighted F1", "Weighted\nPrecision", "Weighted\nRecall"]

    vals = np.array([[results[m][k] for k in metric_keys] for m in MODELS])
    errs = ci95(vals, N_TOTAL)

    x     = np.arange(len(metric_keys))
    width = 0.22

    # GridSpec: chart row + table row
    fig = plt.figure(figsize=(10, 7))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.55)
    ax  = fig.add_subplot(gs[0])
    tax = fig.add_subplot(gs[1])
    tax.axis("off")

    for i, (mlabel, color) in enumerate(zip(MODEL_LABELS, MODEL_COLORS)):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, vals[i], width,
            color=color, label=mlabel, alpha=0.88,
            yerr=errs[i], capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#333"},
            zorder=3,
        )
        for bar, v in zip(bars, vals[i]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#222")

    # Star for best per metric
    for j in range(len(metric_keys)):
        best = int(np.argmax(vals[:, j]))
        offset = (best - 1) * width
        top = vals[best, j] + errs[best, j] + 0.013
        ax.text(x[j] + offset, top, "★", ha="center",
                fontsize=10, color=MODEL_COLORS[best])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, ha="center")
    ax.set_ylim(0.84, 1.00)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score")
    ax.set_title("NeuroScan: Overall Model Performance  (test set, n = 1,600)", pad=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    panel_label(ax, "(a)")

    # Shared legend centred below x-axis, above table (same style as fig 3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", ncol=3,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.14), fontsize=9)

    # Three-line table
    col_labels = ["Model"] + ["Accuracy", "Weighted F1", "W. Precision", "W. Recall"]
    table_data = []
    for i, (mlabel, row) in enumerate(zip(MODEL_LABELS, vals)):
        best_mask = [int(np.argmax(vals[:, j])) == i for j in range(len(metric_keys))]
        table_data.append(
            [mlabel] + [f"{'*' if best_mask[j] else ''}{row[j]:.4f}"
                        for j in range(len(metric_keys))]
        )

    tbl = tax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Three-line style: hide all borders, keep top/header/bottom only
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        if r == 0:
            cell.set_edgecolor("#333")
            cell.set_linewidth(1.2)
            cell.visible_edges = "TB"
        elif r == len(MODELS):
            cell.visible_edges = "B"
            cell.set_edgecolor("#333")
            cell.set_linewidth(1.0)
        else:
            cell.visible_edges = ""
        if r == 0:
            cell.set_text_props(fontweight="bold")

    # Bold best values
    for i, row in enumerate(vals):
        for j in range(len(metric_keys)):
            if int(np.argmax(vals[:, j])) == i:
                tbl[i + 1, j + 1].set_text_props(fontweight="bold")

    tax.set_title("* = best per metric  |  Error bars = 95 % CI (normal approx., n = 1,600)",
                  fontsize=8, color="#555", pad=4)

    save_fig(fig, "fig1_model_overview")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-class recall heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig2_recall_heatmap():
    recall = np.array([
        [results[m]["per_class"][c]["recall"] for c in CLASSES]
        for m in MODELS
    ])
    errs = ci95(recall, N_CLASS)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    im = ax.imshow(recall, cmap="RdYlGn", vmin=0.70, vmax=1.00, aspect="auto")

    for i in range(len(MODELS)):
        for j in range(len(CLASSES)):
            v  = recall[i, j]
            e  = errs[i, j]
            fg = "white" if v < 0.80 or v > 0.96 else "#111"
            ax.text(j, i - 0.10, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=fg)
            ax.text(j, i + 0.22, f"±{e:.3f}",
                    ha="center", va="center",
                    fontsize=8, color=fg, alpha=0.85)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASS_LABELS, fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=10)
    ax.set_title("Per-Class Recall: All Models  (400 samples / class)", pad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Recall", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.text(0.5, -0.04,
             "Recall = TP / (TP + FN).  Error = 95 % CI (normal approx., n = 400 per class).",
             ha="center", fontsize=8, color="#555")

    panel_label(ax, "(b)", x=-0.09)
    save_fig(fig, "fig2_recall_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-class Precision / Recall / F1 (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_per_class_metrics():
    metric_keys   = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    x     = np.arange(len(CLASSES))
    width = 0.24

    for ax, model, mlabel, panel in zip(
            axes, MODELS, MODEL_LABELS, ["(a)", "(b)", "(c)"]):

        for k, (mkey, mk_label, color) in enumerate(
                zip(metric_keys, metric_labels, METRIC_COLORS)):
            vals = np.array([results[model]["per_class"][c][mkey] for c in CLASSES])
            sym_err = ci95(vals, N_CLASS)
            # Clip asymmetrically: upper bound cannot exceed 1.0
            err_lo = sym_err
            err_hi = np.minimum(sym_err, 1.0 - vals)
            offset = (k - 1) * width
            ax.bar(
                x + offset, vals, width,
                color=color, label=mk_label, alpha=0.88,
                yerr=[err_lo, err_hi], capsize=3,
                error_kw={"elinewidth": 1.1, "ecolor": "#333"},
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, fontsize=8.5)
        ax.set_ylim(0.65, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(mlabel, pad=8, fontsize=11)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
        panel_label(ax, panel, x=-0.14)

    axes[0].set_ylabel("Score")
    # Shared legend centred below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04), fontsize=9)

    fig.suptitle("Per-Class Precision, Recall and F1  by Model", fontsize=12, y=1.01)
    fig.text(0.5, -0.10,
             "Error bars = 95 % CI (normal approx., n = 400 per class).\n"
             "CI is exact for recall; approximate for precision and F1 (non-fixed denominators).",
             ha="center", fontsize=8, color="#555")

    save_fig(fig, "fig3_per_class_metrics")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Training curves (3 models × loss + accuracy)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_training_curves():
    panels = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"]]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))

    for row, (model, mlabel, color) in enumerate(
            zip(MODELS, MODEL_LABELS, MODEL_COLORS)):

        history, n1 = load_history(model)
        epochs      = [h["epoch"]     for h in history]
        train_loss  = [h["train_loss"] for h in history]
        val_loss    = [h["val_loss"]   for h in history]
        train_acc   = [h["train_acc"]  for h in history]
        val_acc     = [h["val_acc"]    for h in history]
        split       = n1 + 0.5

        for col, (y_tr, y_va, ylabel, panel) in enumerate([
            (train_loss, val_loss, "Loss",     panels[row][0]),
            (train_acc,  val_acc,  "Accuracy", panels[row][1]),
        ]):
            ax = axes[row][col]
            ax.plot(epochs, y_tr, color=color, lw=1.8, label="Train")
            ax.plot(epochs, y_va, color=color, lw=1.8,
                    ls="--", alpha=0.75, label="Validation")
            ax.axvline(split, color="#999", lw=1.1, ls=":", zorder=0)

            # Phase annotations in axes coordinates
            rel_split = (split - min(epochs)) / (max(epochs) - min(epochs))
            ax.text(rel_split - 0.03, 0.97, "Phase 1",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color="#666")
            ax.text(rel_split + 0.03, 0.97, "Phase 2",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color="#666")

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            if col == 1:
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1, decimals=0))
            ax.legend(fontsize=8, framealpha=0.85)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{mlabel}: {ylabel}", pad=7, fontsize=10)
            panel_label(ax, panel, x=-0.16)

    fig.suptitle(
        "Training Curves: Loss and Accuracy  (Phase 1: frozen backbone;  Phase 2: fine-tuned)",
        fontsize=11, y=1.005,
    )
    fig.text(
        0.5, -0.015,
        "Solid = train,  Dashed = validation.  Dotted line separates Phase 1 from Phase 2.\n"
        "DenseNet121 uses focal loss (γ = 2.0, class-weighted); loss values not comparable "
        "across models.",
        ha="center", fontsize=8, color="#555",
    )
    fig.tight_layout()
    save_fig(fig, "fig4_training_curves")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating NeuroScan figures...\n")
    print("Figure 1 — Model overview")
    fig1_model_overview()
    print("Figure 2 — Recall heatmap")
    fig2_recall_heatmap()
    print("Figure 3 — Per-class metrics")
    fig3_per_class_metrics()
    print("Figure 4 — Training curves")
    fig4_training_curves()
    print(f"\nDone. Figures saved to: {FIG_DIR}")
