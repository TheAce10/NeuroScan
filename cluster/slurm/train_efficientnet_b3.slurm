#!/bin/bash
#SBATCH --job-name=neuroscan_effb3      # Job name shown in squeue
#SBATCH --partition=gpu-classroom       # Classroom GPU partition (A10G)
#SBATCH --gres=gpu:1                    # Request 1 GPU (single A10G, 24GB)
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --mem=32G                       # System RAM
#SBATCH --time=03:30:00                 # Wall time (must be <= 4h)
#SBATCH --output=logs/effb3_%j.out      # stdout — %j = job ID
#SBATCH --error=logs/effb3_%j.err       # stderr

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/12.4                   # Load CUDA 12.4 as provided by IT
conda activate neuroscan                # Activate project conda environment

echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started  : $(date)"
echo "=============================="

# ── Paths — edit NEUROSCAN_ROOT to your home directory ───────────────────────
NEUROSCAN_ROOT="$HOME/neuroscan"
DATA_DIR="$NEUROSCAN_ROOT/data/brain-tumor-mri"    # dataset downloaded by kaggle_setup.sh
OUTPUT_DIR="$NEUROSCAN_ROOT/weights/efficientnet_b3"

mkdir -p "$OUTPUT_DIR" logs

# ── Train EfficientNetB3 ──────────────────────────────────────────────────────
# Phase 1: 10 epochs frozen backbone  (~15 min on A10G)
# Phase 2: 15 epochs unfrozen last 3 feature blocks (~25 min on A10G)
python "$NEUROSCAN_ROOT/cluster/src/train.py" \
    --model        efficientnet_b3 \
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
echo "Weights  : $OUTPUT_DIR/outputs/efficientnet_b3_neuroscan.pt"
echo "=============================="
