#!/usr/bin/env bash
# NeuroScan — Cluster Environment Setup
# Run this once on the login node after SSH-ing in.
#
# Usage:
#   export KAGGLE_TOKEN=KGAT_ed26f5db1df6e18da9e4b8cb8293497d
#   export WANDB_API_KEY=wandb_v1_aB8rF5XXDeLK9Hjb...
#   bash setup_env.sh
#
# Kaggle credentials: kaggle.com → Account → Settings → API → Create New Token
# The token format is KGAT_... (new Kaggle API format, no separate username needed)
# Kaggle username for this project: blesselikem

set -e  # stop on any error

KAGGLE_USERNAME="blesselikem"

echo "========================================"
echo "  NeuroScan — Cluster Setup"
echo "========================================"

# ── 1. Validate required environment variables ───────────────────────────────
if [[ -z "$KAGGLE_TOKEN" ]]; then
    echo "ERROR: KAGGLE_TOKEN must be exported before running."
    echo "  export KAGGLE_TOKEN=KGAT_..."
    exit 1
fi

if [[ -z "$WANDB_API_KEY" ]]; then
    echo "ERROR: WANDB_API_KEY must be exported before running."
    echo "  export WANDB_API_KEY=your_wandb_api_key"
    exit 1
fi

# ── 2. Install Miniconda (if not already installed) ──────────────────────────
if ! command -v conda &> /dev/null; then
    echo ""
    echo "[1/7] Installing Miniconda..."
    # Use py38_4.12.0 — last release compatible with GLIBC 2.26 (Amazon Linux 2)
    # The base Python (3.8) is only for conda itself; the neuroscan env will use Python 3.11
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "      Miniconda installed at $HOME/miniconda3"
else
    echo "[1/7] Miniconda already installed — skipping."
    eval "$(conda shell.bash hook)"
fi

# ── 3. Create the neuroscan conda environment ────────────────────────────────
echo ""
echo "[2/7] Creating conda environment 'neuroscan' (Python 3.11)..."
if conda env list | grep -q "^neuroscan"; then
    echo "      Environment already exists — skipping creation."
else
    conda create -y -n neuroscan python=3.11
fi
conda activate neuroscan

# ── 4. Install Python packages ───────────────────────────────────────────────
echo ""
echo "[3/7] Installing packages..."
# The cluster has GLIBC 2.26 and GCC 7.3.1. Modern pip packages (scikit-learn, numpy 2.x)
# only ship manylinux_2_28 wheels (requires GLIBC 2.28+) and fall back to building from
# source, which then needs GCC >= 9.3. conda-forge builds against the older manylinux_2_17
# standard, so we use conda for the scientific stack and pip only for pure-Python packages.

echo "      [3a] pip: scientific stack with manylinux_2_17-compatible version pins..."
# Version pins: cluster has GLIBC 2.26 and GCC 7.3.1.
# Packages are pinned to the last release that ships manylinux_2_17 pre-built wheels.
# Anything newer only ships manylinux_2_28 wheels and falls back to source,
# which then requires GCC >= 9.3 (cluster has 7.3.1) and fails.
pip install --quiet \
    "numpy<2.0" \
    "scipy<1.15" \
    "pandas<2.3" \
    "scikit-learn<1.6" \
    "matplotlib<3.10" \
    seaborn \
    Pillow

echo "      [3b] pip: PyTorch CPU build..."
# CPU-only PyTorch build — works on compute nodes without GPUs.
# When GPU nodes become available, reinstall with the CUDA build:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "      [3c] pip: pure-Python packages (kaggle, wandb, tqdm)..."
pip install --quiet \
    kaggle \
    "wandb<0.16.0" \
    "setuptools<70" \
    tqdm

echo "      Packages installed."

# ── 5. Set up Kaggle credentials ─────────────────────────────────────────────
# The new Kaggle API token format (KGAT_...) uses a single token field.
# The Kaggle CLI reads ~/.kaggle/kaggle.json at startup.
echo ""
echo "[4/7] Configuring Kaggle API credentials..."
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_TOKEN}"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"
echo "      Kaggle credentials saved to ~/.kaggle/kaggle.json (username: ${KAGGLE_USERNAME})"

# ── 6. Download and extract the Brain Tumor dataset ──────────────────────────
DATASET_DIR="/shared/scratch/group2/brain-tumor-mri"

echo ""
echo "[5/7] Downloading Brain Tumor MRI dataset from Kaggle..."
if [ -d "$DATASET_DIR/Training" ] && [ -d "$DATASET_DIR/Testing" ]; then
    echo "      Dataset already exists at $DATASET_DIR — skipping download."
else
    mkdir -p "$DATASET_DIR"
    # Download to a temporary location then extract into the shared datasets folder
    TMPZIP="/tmp/brain-tumor-mri-dataset.zip"
    kaggle datasets download \
        -d masoudnickparvar/brain-tumor-mri-dataset \
        -p /tmp --quiet
    echo "      Extracting dataset to $DATASET_DIR ..."
    unzip -q "$TMPZIP" -d "$DATASET_DIR"
    rm "$TMPZIP"
    echo "      Dataset extracted."
fi

# Verify structure
echo "      Dataset contents:"
ls "$DATASET_DIR"

# ── 7. Create group scratch directory ────────────────────────────────────────
echo ""
echo "[6/7] Creating Group 2 scratch directories..."
SCRATCH_DIR="/shared/scratch/group2/neuroscan"
mkdir -p "$SCRATCH_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR/logs"
mkdir -p "$SCRATCH_DIR/outputs"
echo "      Created: $SCRATCH_DIR"
echo "               $SCRATCH_DIR/checkpoints"
echo "               $SCRATCH_DIR/logs"
echo "               $SCRATCH_DIR/outputs"

# ── 8. Configure Weights and Biases ──────────────────────────────────────────
# The wandb CLI login is skipped because the 0.15.x CLI has a pkg_resources
# compatibility issue on this cluster. The Python API (used in train.py) reads
# WANDB_API_KEY directly from the environment — no CLI login needed.
echo ""
echo "[7/7] Configuring Weights and Biases API key..."
if ! grep -q "WANDB_API_KEY" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# Weights and Biases API key — added by NeuroScan setup" >> "$HOME/.bashrc"
    echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> "$HOME/.bashrc"
fi
echo "      WANDB_API_KEY written to ~/.bashrc"
echo "      Verifying Python can import wandb..."
python -c "import wandb; print('      wandb', wandb.__version__, '— import OK')"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Dataset : $DATASET_DIR"
echo "  Scratch : $SCRATCH_DIR"
echo "  Env     : conda activate neuroscan"
echo ""
echo "  Next: sbatch submit.slurm"
echo "========================================"
