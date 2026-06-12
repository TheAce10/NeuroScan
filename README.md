# NeuroScan

Brain tumour MRI classification using transfer learning. An ensemble of three CNN models classifies MRI scans into glioma, meningioma, pituitary tumour, or no tumour — with Grad-CAM heatmap visualisation.

**Live demo:** *(coming soon)*

---

## Model Performance

| Model | Accuracy | Glioma Recall | Meningioma Recall |
|---|---|---|---|
| EfficientNetB3 | 88.4% | 71% | 87% |
| VGG16 | 88.9% | 76% | 82% |
| **DenseNet121** (focal loss) | **91.8%** | **79%** | **94%** |
| **Ensemble** | — | — | — |

All models trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) via Kaggle T4 GPU.

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
