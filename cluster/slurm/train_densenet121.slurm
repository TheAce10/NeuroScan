#!/bin/bash
#SBATCH --job-name=neuroscan_dense121   # Job name shown in squeue
#SBATCH --partition=gpu-classroom       # Classroom GPU partition (A10G)
#SBATCH --gres=gpu:1                    # Request 1 GPU (single A10G, 24GB)
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --mem=32G                       # System RAM
#SBATCH --time=03:30:00                 # Wall time (must be <= 4h)
#SBATCH --output=logs/densenet_%j.out   # stdout — %j = job ID
#SBATCH --error=logs/densenet_%j.err    # stderr

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/12.4
conda activate neuroscan

echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started  : $(date)"
echo "=============================="

# ── Paths — edit NEUROSCAN_ROOT to your home directory ───────────────────────
NEUROSCAN_ROOT="$HOME/neuroscan"
DATA_DIR="$NEUROSCAN_ROOT/data/brain-tumor-mri"
OUTPUT_DIR="$NEUROSCAN_ROOT/weights/densenet121"

mkdir -p "$OUTPUT_DIR" logs

# ── Train DenseNet121 ─────────────────────────────────────────────────────────
# Uses FocalLoss (gamma=2.0) + class weights [2.0, 1.2, 0.8, 0.8] to address
# glioma/meningioma class imbalance — best model for minority-class recall.
# Phase 1: 10 epochs frozen backbone  (~12 min on A10G)
# Phase 2: 15 epochs unfrozen denseblock4 + norm5  (~20 min on A10G)
python "$NEUROSCAN_ROOT/cluster/src/train.py" \
    --model        densenet121 \
    --data-dir     "$DATA_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    --epochs-phase1  10 \
    --epochs-phase2  15 \
    --batch-size     32 \
    --lr-phase1      1e-3 \
    --lr-phase2      5e-5 \
    --num-workers    8

echo "=============================="
echo "Finished : $(date)"
echo "Weights  : $OUTPUT_DIR/outputs/densenet121_neuroscan.pt"
echo "=============================="
