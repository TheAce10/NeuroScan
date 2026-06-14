# Model Weights

Weights are **not tracked in git** — download from Hugging Face Hub.

## Hugging Face Hub

Repository: https://huggingface.co/The-Ace-000/neuroscan-weights

| File                          | Model          | Accuracy |
|-------------------------------|----------------|----------|
| `efficientnet_b3_neuroscan.pt`| EfficientNetB3 | 92.0%    |
| `vgg16_neuroscan.pt`          | VGG16          | 93.0%    |
| `densenet121_neuroscan.pt`    | DenseNet121    | 91.0%    |

## Local structure (for running backend locally)

```
weights/neuroscan/
  efficientnet_b3/outputs/efficientnet_b3_neuroscan.pt
  vgg16/outputs/vgg16_neuroscan.pt
  densenet121/outputs/densenet121_neuroscan.pt
```

The backend (`main.py`) will auto-download from HF Hub if local files are not present.
