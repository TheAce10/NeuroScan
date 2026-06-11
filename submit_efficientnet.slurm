#!/bin/bash
#SBATCH --job-name=ns-efficientnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/elikem/group2/neuroscan/efficientnet_b3/logs/run_%j.out
#SBATCH --error=/shared/home/elikem/group2/neuroscan/efficientnet_b3/logs/run_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=krapahbless@gmail.com

echo "============================================"
echo "  NeuroScan — EfficientNetB3"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Started  : $(date)"
echo "============================================"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate neuroscan

python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

mkdir -p /shared/home/elikem/group2/neuroscan/efficientnet_b3/{logs,checkpoints,outputs}

python "$HOME/group2/train.py" \
    --model          efficientnet_b3 \
    --data-dir       /shared/home/elikem/group2/brain-tumor-mri \
    --output-dir     /shared/home/elikem/group2/neuroscan/efficientnet_b3 \
    --epochs-phase1  10 \
    --epochs-phase2  10 \
    --lr-phase1      1e-3 \
    --lr-phase2      5e-5 \
    --batch-size     32 \
    --num-workers    4

EXIT_CODE=$?
echo "============================================"
echo "  Finished : $(date)  |  Exit: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
