# NeuroScan — Cluster Training Guide

This guide walks a new user through reproducing NeuroScan's brain tumor MRI
classification models on the department HPC cluster. It covers environment
setup, dataset download via Kaggle CLI, and SLURM job submission.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Models](#2-models)
3. [Cluster Specifications](#3-cluster-specifications)
4. [Initial Setup](#4-initial-setup)
5. [Kaggle API Setup](#5-kaggle-api-setup)
6. [Dataset Download](#6-dataset-download)
7. [Conda Environment](#7-conda-environment)
8. [Repository Structure](#8-repository-structure)
9. [Running Training Jobs](#9-running-training-jobs)
10. [Running Evaluation](#10-running-evaluation)
11. [Submit All Jobs at Once](#11-submit-all-jobs-at-once)
12. [Monitoring Jobs](#12-monitoring-jobs)
13. [Output Files](#13-output-files)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

**NeuroScan** is a brain tumor classification system trained on MRI scans.
It classifies each scan into one of four categories:

| Class | Description |
|---|---|
| `glioma` | Glioma tumour |
| `meningioma` | Meningioma tumour |
| `pituitary` | Pituitary tumour |
| `notumor` | No tumour detected |

The dataset is the
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar on Kaggle (7,023 MRI scans).

All four models use **two-phase transfer learning**:
- **Phase 1** — Backbone frozen, only the custom head trains (10 epochs)
- **Phase 2** — Last convolutional blocks unfrozen, end-to-end fine-tuning (15 epochs)

Both phases use **AdamW** optimiser with **CosineAnnealingLR** scheduling.

---

## 2. Models

### EfficientNetB3
- **Input:** 224 × 224
- **Head:** `Dropout(0.3) → Linear(1536, 4)`
- **Phase 2 unfrozen:** `features.6`, `features.7`, `features.8`
- **Loss:** CrossEntropyLoss

### VGG16
- **Input:** 224 × 224
- **Head:** Original classifier layers 0–5 + `Dropout(0.4) → Linear(4096, 4)`
- **Phase 2 unfrozen:** `features[24:]` (last Conv512 block)
- **Loss:** CrossEntropyLoss

### DenseNet121
- **Input:** 224 × 224
- **Head:** `Dropout(0.4) → Linear(1024, 4)`
- **Phase 2 unfrozen:** `denseblock4`, `norm5`
- **Loss:** FocalLoss (γ=2.0) + class weights `[2.0, 1.2, 0.8, 0.8]`
- **Note:** FocalLoss and upweighted glioma address the glioma/meningioma
  confusion seen in baseline cross-entropy models.

### InceptionV3
- **Input:** 299 × 299
- **Head:** `Dropout(0.3) → Linear(2048, 4)`; auxiliary head also fine-tuned
- **Phase 1:** Auxiliary loss weighted at 0.4× (standard Inception training)
- **Phase 2 unfrozen:** `Mixed_7a`, `Mixed_7b`, `Mixed_7c`
- **Loss:** CrossEntropyLoss

---

## 3. Cluster Specifications

| Resource | Value |
|---|---|
| GPU | NVIDIA A10G (24 GB Ampere) |
| Partition | `gpu-classroom` |
| CUDA version | 12.4 |
| Max wall time | **4 hours** per job |
| Cards per job | 1 (single GPU) |

> **Important:** The `--time` in all SLURM scripts is set to `03:30:00`
> (training) and `01:00:00` (evaluation) to stay within the 4-hour limit
> with a safety buffer.

---

## 4. Initial Setup

Log in to the cluster:

```bash
# Replace <username> with your cluster username
ssh <username>@<cluster-hostname>
```

Clone the repository and set it up in your home directory:

```bash
# Clone the NeuroScan repository into ~/neuroscan
git clone <repository-url> ~/neuroscan

# Set up the directory structure expected by the SLURM scripts
mkdir -p ~/neuroscan/data          # dataset will go here
mkdir -p ~/neuroscan/weights       # model weights output
mkdir -p ~/neuroscan/results       # evaluation output
mkdir -p ~/neuroscan/logs          # SLURM log files
```

---

## 5. Kaggle API Setup

You need a Kaggle API token to download the dataset. Follow these steps.

### Step 1 — Generate your Kaggle token

1. Log in at [https://www.kaggle.com](https://www.kaggle.com)
2. Click your profile picture (top-right) → **Settings**
3. Scroll to **API** section → click **Create New Token**
4. A file called `kaggle.json` will download to your computer.
   It contains:
   ```json
   {"username":"your-kaggle-username","key":"your-api-key-here"}
   ```

> **Keep this token secret.** Do not commit it to git or share it.

### Step 2 — Transfer the token to the cluster

From your **local machine**, copy the token to the cluster:

```bash
# Create the .kaggle directory on the cluster
ssh <username>@<cluster-hostname> "mkdir -p ~/.kaggle"

# Copy your token file to the cluster
scp ~/Downloads/kaggle.json <username>@<cluster-hostname>:~/.kaggle/kaggle.json
```

### Step 3 — Set permissions on the cluster

SSH into the cluster and secure the token file:

```bash
# The Kaggle CLI requires the file to be readable only by you
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4 — Export credentials as environment variables (cluster sessions)

Some cluster environments do not preserve `~/.kaggle/kaggle.json` across
interactive and batch sessions. Export the credentials explicitly:

```bash
# Add these two lines to your ~/.bashrc so they load in every session
echo 'export KAGGLE_USERNAME="your-kaggle-username"' >> ~/.bashrc
echo 'export KAGGLE_KEY="your-api-key-here"'         >> ~/.bashrc

# Reload .bashrc in the current session
source ~/.bashrc
```

Replace `your-kaggle-username` and `your-api-key-here` with the values from
`kaggle.json`.

### Step 5 — Test the Kaggle CLI

```bash
# Verify the CLI can authenticate
kaggle datasets list --search "brain-tumor-mri"
# Expected: a table listing the dataset masoudnickparvar/brain-tumor-mri-dataset
```

---

## 6. Dataset Download

```bash
# Download and unzip the Brain Tumor MRI Dataset into the data directory
# Dataset slug: masoudnickparvar/brain-tumor-mri-dataset
kaggle datasets download \
    masoudnickparvar/brain-tumor-mri-dataset \
    --path ~/neuroscan/data \
    --unzip

# After unzipping, verify the directory structure:
ls ~/neuroscan/data/brain-tumor-mri/
# Expected output:
#   Training/
#   Testing/

ls ~/neuroscan/data/brain-tumor-mri/Training/
# Expected output:
#   glioma/  meningioma/  notumor/  pituitary/

# Check sample counts (should be ~5,712 training, ~1,311 testing)
find ~/neuroscan/data/brain-tumor-mri/Training -name "*.jpg" | wc -l
find ~/neuroscan/data/brain-tumor-mri/Testing  -name "*.jpg" | wc -l
```

> If the unzipped folder is named differently (e.g., `Brain Tumor MRI Dataset/`),
> rename it:
> ```bash
> mv ~/neuroscan/data/"Brain Tumor MRI Dataset" ~/neuroscan/data/brain-tumor-mri
> ```

---

## 7. Conda Environment

Load the Anaconda module and create the environment:

```bash
# Load the cluster's conda module (module name may differ — check with: module avail)
module load anaconda3
# or: module load miniconda3

# Create the neuroscan environment from the spec file
# This installs PyTorch 2.3+ with CUDA 12.4, scikit-learn, matplotlib, Kaggle CLI, etc.
conda env create -f ~/neuroscan/cluster/environment.yml

# Activate the environment
conda activate neuroscan

# Verify PyTorch can see the GPU (run this on a GPU node via srun)
srun --partition=gpu-classroom --gres=gpu:1 --pty bash -c \
    "conda activate neuroscan && python -c 'import torch; print(torch.cuda.get_device_name(0))'"
# Expected output: NVIDIA A10G
```

> **Note:** Building the conda environment from scratch can take 5–10 minutes
> due to PyTorch download size (~2 GB). Run it once and reuse across jobs.

---

## 8. Repository Structure

After cloning, the relevant files are:

```
neuroscan/
├── cluster/
│   ├── README.md                   ← this file
│   ├── environment.yml             ← conda environment spec
│   ├── src/
│   │   ├── train.py                ← training script (all 4 models)
│   │   └── eval_compare.py         ← evaluation + plots script
│   └── slurm/
│       ├── train_efficientnet_b3.slurm
│       ├── train_vgg16.slurm
│       ├── train_densenet121.slurm
│       ├── train_inception_v3.slurm
│       ├── eval_compare.slurm
│       └── submit_all.sh           ← submit all jobs in one command
├── data/
│   └── brain-tumor-mri/            ← dataset (downloaded in step 6)
│       ├── Training/
│       └── Testing/
├── weights/                        ← model weights (created by training)
│   ├── efficientnet_b3/outputs/efficientnet_b3_neuroscan.pt
│   ├── vgg16/outputs/vgg16_neuroscan.pt
│   ├── densenet121/outputs/densenet121_neuroscan.pt
│   └── inception_v3/outputs/inception_v3_neuroscan.pt
├── results/                        ← evaluation outputs (created by eval job)
│   ├── model_comparison.json
│   ├── model_comparison.csv
│   └── figures/
│       ├── confusion_efficientnet_b3.png
│       ├── confusion_vgg16.png
│       ├── confusion_densenet121.png
│       ├── confusion_inception_v3.png
│       ├── accuracy_comparison.png
│       ├── per_class_recall.png
│       └── per_class_f1.png
└── logs/                           ← SLURM stdout/stderr logs
```

---

## 9. Running Training Jobs

### Option A — Submit jobs individually

```bash
cd ~/neuroscan/cluster/slurm

# Submit EfficientNetB3 (~3.5h wall time)
sbatch train_efficientnet_b3.slurm

# Submit VGG16 (~3.5h wall time — heavier than EfficientNet)
sbatch train_vgg16.slurm

# Submit DenseNet121 (~3h wall time)
sbatch train_densenet121.slurm

# Submit InceptionV3 (~3.5h wall time)
sbatch train_inception_v3.slurm
```

### Option B — Submit all jobs at once (recommended)

See [Section 11](#11-submit-all-jobs-at-once).

### What each training job does

1. Loads the dataset from `~/neuroscan/data/brain-tumor-mri/`
2. Builds the model with pretrained ImageNet weights
3. **Phase 1** (10 epochs): trains only the classification head
4. **Phase 2** (15 epochs): unfreezes last conv blocks, fine-tunes end-to-end
5. Saves the best-validation-accuracy weights to `~/neuroscan/weights/<model>/outputs/`
6. Saves per-epoch history to `~/neuroscan/weights/<model>/checkpoints/phase{1,2}_history.json`
7. Saves a training summary to `~/neuroscan/weights/<model>/training_summary.json`

---

## 10. Running Evaluation

After all training jobs finish, run the evaluation job:

```bash
cd ~/neuroscan/cluster/slurm

# Submit the evaluation job
sbatch eval_compare.slurm
```

The evaluation job:
1. Discovers all weight files under `~/neuroscan/weights/`
2. Evaluates each model on the full test set (400 × 4 = 1,600 samples)
3. Computes accuracy, F1, precision, recall (overall and per class)
4. Generates confusion matrix PNGs for each model
5. Generates accuracy comparison, per-class recall, and per-class F1 bar charts
6. Plots training curves (Phase 1 + Phase 2 combined) for each model
7. Saves `model_comparison.json` and `model_comparison.csv` to `~/neuroscan/results/`

### Copy results to your local machine

```bash
# From your local machine, copy all results and figures
scp -r <username>@<cluster-hostname>:~/neuroscan/results/ ./neuroscan_results/
```

---

## 11. Submit All Jobs at Once

The `submit_all.sh` script submits all five jobs with proper dependencies.
All four training jobs run concurrently. The evaluation job only starts after
every training job has completed successfully.

```bash
cd ~/neuroscan/cluster/slurm

# Make the script executable (only needed once)
chmod +x submit_all.sh

# Submit the full pipeline
bash submit_all.sh
```

Example output:

```
===================================================
  NeuroScan — Submitting SLURM job chain
  Fri Jun 13 10:00:00 UTC 2026
===================================================
Submitted EfficientNetB3  : Job 100042
Submitted VGG16           : Job 100043
Submitted DenseNet121     : Job 100044
Submitted InceptionV3     : Job 100045
Submitted Evaluation      : Job 100046  (depends on 100042, 100043, 100044, 100045)

===================================================
  Job chain submitted successfully.

  Monitor:    squeue -u $USER
  Cancel all: scancel 100042 100043 100044 100045 100046
===================================================

  Expected total wall time: ~3.5h (training) + ~0.5h (eval)
  Results will appear in:   ~/neuroscan/results/
```

The job dependency (`--dependency=afterok`) means:
- If any training job **fails**, the evaluation job is automatically **cancelled**.
- If all training jobs succeed, evaluation starts immediately.

---

## 12. Monitoring Jobs

```bash
# Show your current jobs and their state
squeue -u $USER

# Show your job history (recent completed/failed jobs)
sacct -u $USER --format=JobID,JobName,State,Elapsed,Start,End

# Stream a job's log in real time (replace <JOBID> with your job number)
tail -f ~/neuroscan/logs/effb3_<JOBID>.out

# Show GPU utilisation on a running node (find node name from squeue)
ssh <node-name> nvidia-smi

# Cancel a specific job
scancel <JOBID>

# Cancel ALL your jobs
scancel -u $USER
```

**SLURM job states:**
| State | Meaning |
|---|---|
| `PD` | Pending — waiting for a free GPU |
| `R` | Running |
| `CG` | Completing |
| `CD` | Completed successfully |
| `F` | Failed — check the `.err` log |
| `CA` | Cancelled |

---

## 13. Output Files

After a successful run, you will have:

```
~/neuroscan/weights/efficientnet_b3/
    outputs/efficientnet_b3_neuroscan.pt     ← final weights (best val acc)
    checkpoints/phase1_best.pt               ← best checkpoint in Phase 1
    checkpoints/phase2_best.pt               ← best checkpoint in Phase 2
    checkpoints/phase1_history.json          ← per-epoch loss/acc (Phase 1)
    checkpoints/phase2_history.json          ← per-epoch loss/acc (Phase 2)
    training_summary.json                    ← final test metrics + hyperparams

# (same layout for vgg16, densenet121, inception_v3)

~/neuroscan/results/
    model_comparison.json    ← full metrics for all models
    model_comparison.csv     ← flat table for spreadsheets
    figures/
        confusion_efficientnet_b3.png
        confusion_vgg16.png
        confusion_densenet121.png
        confusion_inception_v3.png
        training_curve_efficientnet_b3.png
        training_curve_vgg16.png
        training_curve_densenet121.png
        training_curve_inception_v3.png
        accuracy_comparison.png
        per_class_recall.png
        per_class_f1.png
```

The `model_comparison.json` structure:

```json
{
  "efficientnet_b3": {
    "accuracy": 0.9175,
    "f1_weighted": 0.9180,
    "precision_weighted": 0.9190,
    "recall_weighted": 0.9175,
    "per_class": {
      "glioma":      {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 300},
      "meningioma":  {"precision": 0.87, "recall": 0.90, "f1": 0.88, "support": 306},
      "notumor":     {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 405},
      "pituitary":   {"precision": 0.95, "recall": 0.96, "f1": 0.96, "support": 300}
    }
  },
  "vgg16":      { ... },
  "densenet121": { ... },
  "inception_v3": { ... }
}
```

---

## 14. Troubleshooting

### Job fails immediately with `ModuleNotFoundError: No module named 'torch'`
The conda environment is not activated inside the SLURM script.
Make sure `conda activate neuroscan` runs **after** `module load anaconda3`.
Some clusters require `source activate neuroscan` instead:

```bash
# In the SLURM script, replace:
conda activate neuroscan
# with:
source activate neuroscan
# or the full path:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate neuroscan
```

### `conda activate` not found in batch job
Add this before the activate call:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuroscan
```

### Job hits the wall-time limit (state `TO` = timeout)
Training of heavier models (VGG16, InceptionV3) with large batch sizes may
run close to 3.5 hours. If jobs are timing out, reduce batch size:

```bash
# In the SLURM script, change:
--batch-size 32
# to:
--batch-size 16
```

### `CUDA out of memory`
The A10G has 24 GB. This should be sufficient for batch=32.
If you see OOM, reduce batch size to 16 and re-submit.

### Kaggle download fails with `401 - Unauthorized`
The API credentials are not set correctly. Verify:

```bash
# Check the file exists and is readable
cat ~/.kaggle/kaggle.json
# Should print: {"username":"...","key":"..."}

# Check environment variables are set
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY
```

If environment variables are missing, re-run the export commands in
[Section 5 Step 4](#step-4--export-credentials-as-environment-variables-cluster-sessions)
and `source ~/.bashrc`.

### Dataset directory not found by training script
Verify the path passed to `--data-dir` contains `Training/` and `Testing/`
subdirectories, each with four class folders (`glioma`, `meningioma`,
`notumor`, `pituitary`):

```bash
ls ~/neuroscan/data/brain-tumor-mri/Training/
# glioma  meningioma  notumor  pituitary
```

### Evaluation job cancelled automatically
This means one or more training jobs failed (state `F`).
Check the `.err` logs:

```bash
cat ~/neuroscan/logs/effb3_<JOBID>.err
```

Fix the issue, re-run the failed training jobs individually, then re-submit
the evaluation job manually:

```bash
sbatch ~/neuroscan/cluster/slurm/eval_compare.slurm
```

### `ImportError: libcuda.so.1: cannot open shared object file`
CUDA libraries are not loaded. Make sure `module load cuda/12.4` appears in
your SLURM script **before** `conda activate`.

---

## Quick Reference

```bash
# Full pipeline in three steps

# 1. Download dataset
kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset \
    --path ~/neuroscan/data --unzip

# 2. Create environment (once)
conda env create -f ~/neuroscan/cluster/environment.yml

# 3. Submit all jobs
bash ~/neuroscan/cluster/slurm/submit_all.sh
```
