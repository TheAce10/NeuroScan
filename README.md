# NeuroScan

Brain tumour MRI classification using transfer learning. An ensemble of three CNN models classifies MRI scans into glioma, meningioma, pituitary tumour, or no tumour — with Grad-CAM heatmap visualisation.

**Live demo:** https://theace10.github.io/NeuroScan/

---

## Model Performance

Evaluated on 1,600 balanced test samples (400 per class). All models trained with skull-strip preprocessing.

| Model | Accuracy | F1 (weighted) | Glioma Recall | Meningioma Recall |
|---|---|---|---|---|
| **VGG16** | **93.0%** | **92.9%** | **79.3%** | **94.8%** |
| EfficientNetB3 | 92.0% | 91.8% | 77.8% | 91.8% |
| DenseNet121 | 91.0% | 90.9% | 80.8% | 91.5% |

All models trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 MRI scans · 5,712 training / 1,311 test · 4 classes).

---

## Repository Structure

```
NeuroScan/
  src/
    training/         Training scripts and SLURM cluster jobs
    web/              React frontend + FastAPI backend
  notebooks/          Kaggle training notebooks (EfficientNetB3+VGG16, DenseNet121)
  data/               README with dataset and weights download links
  results/            Evaluation outputs and classification reports
  figures/            Confusion matrices and accuracy plots
  docs/               Architecture and API documentation
  tests/              Backend API tests
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate neuroscan

# Download weights (see data/README.md)

# Start backend
cd src/web/neuroscan-web/backend
uvicorn main:app --reload

# Start frontend (separate terminal)
cd src/web/neuroscan-web
npm install && npm run dev
```

Open http://localhost:5173

## Research Goal

Assess the readiness of AI/ML to contribute to medical expertise — specifically whether transfer learning on public MRI datasets can reach clinically useful accuracy on tumour type classification, and where model uncertainty is highest (glioma/meningioma boundary cases).

## Citation

See [CITATION.cff](CITATION.cff).
