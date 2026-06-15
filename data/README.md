# Data

Training and test images are **not tracked in git** — download from Kaggle.

**Brain Tumor MRI Dataset** — Masoud Nickparvar  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

| Split    | Images per class | Total |
|----------|-----------------|-------|
| Training | 1,400           | 5,600 |
| Testing  | 400             | 1,600 |

Classes: `glioma` · `meningioma` · `pituitary` · `notumor`

## Local structure (after download)

```
data/
  Training/
    glioma/         (1,400 images)
    meningioma/     (1,400 images)
    notumor/        (1,400 images)
    pituitary/      (1,400 images)
  Testing/
    glioma/         (400 images)
    meningioma/     (400 images)
    notumor/        (400 images)
    pituitary/      (400 images)
```

See `docs/documentation.md` for full dataset rationale and preprocessing pipeline.
