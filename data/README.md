# Data

This directory is intentionally empty. Model weights and dataset are not tracked in git.

## Dataset

**Brain Tumor MRI Dataset** — Masoud Nickparvar
Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, pituitary, no tumour
Train: 5,712 images | Test: 1,311 images

## Model Weights

Hosted on Hugging Face Hub: *(link to be added after upload)*

Expected local structure for running the backend:
```
weights/neuroscan/
  efficientnet_b3/outputs/efficientnet_b3_neuroscan.pt
  vgg16/checkpoints/phase2_best.pt
  densenet121/outputs/densenet121_neuroscan.pt
```
