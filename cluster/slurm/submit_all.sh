#!/bin/bash
# submit_all.sh — Submit all NeuroScan SLURM jobs in dependency order.
#
# Jobs run concurrently where possible (all 4 training jobs in parallel).
# Evaluation only starts after ALL training jobs have succeeded.
#
# Usage:
#   cd ~/neuroscan/cluster/slurm
#   bash submit_all.sh
#
# After submitting, monitor with:
#   watch squeue -u $USER

set -euo pipefail   # exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================================="
echo "  NeuroScan — Submitting SLURM job chain"
echo "  $(date)"
echo "==================================================="

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/../logs"
cd "$SCRIPT_DIR"   # make --output/--error relative paths work

# ── Submit all 4 training jobs (they run concurrently on whatever GPUs are available) ──

JOB_EFFB3=$(sbatch --parsable train_efficientnet_b3.slurm)
echo "Submitted EfficientNetB3  : Job $JOB_EFFB3"

JOB_VGG16=$(sbatch --parsable train_vgg16.slurm)
echo "Submitted VGG16           : Job $JOB_VGG16"

JOB_DENSE=$(sbatch --parsable train_densenet121.slurm)
echo "Submitted DenseNet121     : Job $JOB_DENSE"

JOB_INC=$(sbatch --parsable train_inception_v3.slurm)
echo "Submitted InceptionV3     : Job $JOB_INC"

# ── Submit evaluation job — only runs after ALL 4 training jobs succeed ──────
# --dependency=afterok:<j1>:<j2>:... means: start only if all listed jobs exit 0.
# If any training job fails, eval is cancelled automatically.

JOB_EVAL=$(sbatch --parsable \
    --dependency=afterok:${JOB_EFFB3}:${JOB_VGG16}:${JOB_DENSE}:${JOB_INC} \
    eval_compare.slurm)
echo "Submitted Evaluation      : Job $JOB_EVAL  (depends on $JOB_EFFB3, $JOB_VGG16, $JOB_DENSE, $JOB_INC)"

echo ""
echo "==================================================="
echo "  Job chain submitted successfully."
echo ""
echo "  Monitor:   squeue -u \$USER"
echo "  Cancel all: scancel $JOB_EFFB3 $JOB_VGG16 $JOB_DENSE $JOB_INC $JOB_EVAL"
echo "==================================================="
echo ""
echo "  Expected total wall time: ~3.5h (training) + ~0.5h (eval)"
echo "  Results will appear in:   ~/neuroscan/results/"
