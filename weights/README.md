# Model Weights

Weights are **not tracked in git** — download from Hugging Face Hub.

## Hugging Face Hub

Repository: https://huggingface.co/The-Ace-000/neuroscan-weights

| File                            | Model           | Accuracy |
|---------------------------------|-----------------|----------|
| `efficientnet_b3_neuroscan.pt`  | EfficientNetB3  | 92.0%    |
| `vgg16_neuroscan.pt`            | VGG16           | 93.0%    |
| `densenet121_neuroscan.pt`      | DenseNet121     | 91.0%    |

## Local structure (for running the backend locally)

```
weights/
  efficientnet_b3/
    checkpoints/          # phase1_history.json, phase2_history.json, *.pt
    outputs/              # efficientnet_b3_neuroscan.pt
  vgg16/
    checkpoints/          # phase1_history.json, phase2_history.json, *.pt
    outputs/              # vgg16_neuroscan.pt
  densenet121/
    checkpoints/          # phase1_history.json, phase2_history.json, *.pt
    outputs/              # densenet121_neuroscan.pt
```

The backend (`src/web/neuroscan-web/backend/main.py`) auto-downloads from HF Hub
if local weight files are not present.
