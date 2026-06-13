#!/bin/bash
#SBATCH --job-name=neuroscan_eval       # Job name shown in squeue
#SBATCH --partition=gpu-classroom       # Classroom GPU partition (A10G)
#SBATCH --gres=gpu:1                    # Request 1 GPU for batch inference
#SBATCH --cpus-per-task=8              # CPU cores for data loading + matplotlib
#SBATCH --mem=32G                       # System RAM
#SBATCH --time=01:00:00                 # Evaluation is fast — 1 hour is generous
#SBATCH --output=logs/eval_%j.out       # stdout — %j = job ID
#SBATCH --error=logs/eval_%j.err        # stderr

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
WEIGHTS_DIR="$NEUROSCAN_ROOT/weights"       # script searches all subdirectories
OUTPUT_DIR="$NEUROSCAN_ROOT/results"

mkdir -p "$OUTPUT_DIR" logs

# ── Run evaluation ────────────────────────────────────────────────────────────
# Evaluates all 4 models on the full test set (1600 samples).
# Outputs:
#   results/model_comparison.json   — full per-class metrics
#   results/model_comparison.csv    — flat table
#   results/figures/                — confusion matrices, accuracy bars, recall bars, curves
python "$NEUROSCAN_ROOT/cluster/src/eval_compare.py" \
    --data-dir    "$DATA_DIR" \
    --weights-dir "$WEIGHTS_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --batch-size   32 \
    --num-workers  8

echo "=============================="
echo "Finished : $(date)"
echo "Results  : $OUTPUT_DIR"
echo "Figures  : $OUTPUT_DIR/figures"
echo "=============================="

# Print a quick summary from the JSON output
python - <<'PYEOF'
import json, os
out = os.path.expandvars("$HOME/neuroscan/results/model_comparison.json")
if not os.path.exists(out):
    print("model_comparison.json not found")
    exit(1)
with open(out) as f:
    data = json.load(f)
print("\n=== NEUROSCAN RESULTS SUMMARY ===")
rows = sorted(data.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for mid, m in rows:
    print(f"  {mid:<20}  acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}")
PYEOF
