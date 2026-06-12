import io, base64, os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
DEVICE       = torch.device("cpu")

# When running on HF Spaces, weights are downloaded from HF Hub.
# When running locally, weights are loaded from the local path.
HF_REPO      = "The-Ace-000/neuroscan-weights"
WEIGHTS_ROOT = Path(__file__).resolve().parent.parent.parent / "weights" / "neuroscan"


def resolve_weight(local_path: Path, hf_filename: str) -> Path:
    """Return local path if it exists, otherwise download from HF Hub."""
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import hf_hub_download
        cache = Path(__file__).resolve().parent / "weights_cache"
        cache.mkdir(exist_ok=True)
        downloaded = hf_hub_download(repo_id=HF_REPO, filename=hf_filename,
                                     local_dir=str(cache))
        return Path(downloaded)
    except Exception as e:
        print(f"  [warn] Could not download {hf_filename} from HF Hub: {e}")
        return local_path
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]


# ── model builders ─────────────────────────────────────────────────────────────

def build_efficientnet_b3():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES)),
    )
    return m, m.features[-1]


def build_vgg16():
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(
        *list(m.classifier.children())[:-1],
        nn.Dropout(p=0.4),
        nn.Linear(4096, len(CLASS_NAMES)),
    )
    return m, m.features[28]   # last Conv2d before MaxPool


def build_densenet121():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier.in_features, len(CLASS_NAMES)),
    )
    return m, m.features.denseblock4   # last dense block


MODEL_CONFIGS = {
    "efficientnet_b3": {
        "builder":  build_efficientnet_b3,
        "weights":  resolve_weight(
            WEIGHTS_ROOT / "efficientnet_b3" / "outputs" / "efficientnet_b3_neuroscan.pt",
            "efficientnet_b3_neuroscan.pt",
        ),
        "img_size": 224,
        "display":  "EfficientNetB3",
    },
    "vgg16": {
        "builder":  build_vgg16,
        "weights":  resolve_weight(
            WEIGHTS_ROOT / "vgg16" / "checkpoints" / "phase2_best.pt",
            "vgg16_phase2_best.pt",
        ),
        "img_size": 224,
        "display":  "VGG16",
    },
    "densenet121": {
        "builder":  build_densenet121,
        "weights":  resolve_weight(
            WEIGHTS_ROOT / "densenet121" / "outputs" / "densenet121_neuroscan.pt",
            "densenet121_neuroscan.pt",
        ),
        "img_size": 224,
        "display":  "DenseNet121",
    },
}

loaded: Dict[str, dict] = {}


def try_load(name: str) -> bool:
    cfg = MODEL_CONFIGS[name]
    if not cfg["weights"].exists():
        print(f"  [skip] {name} — weights not found at {cfg['weights']}")
        return False
    try:
        model, target_layer = cfg["builder"]()
        state = torch.load(cfg["weights"], map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        loaded[name] = {
            "model":        model,
            "target_layer": target_layer,
            "img_size":     cfg["img_size"],
            "display":      cfg["display"],
        }
        print(f"  [ok]   {name}")
        return True
    except Exception as e:
        print(f"  [err]  {name}: {e}")
        return False


@asynccontextmanager
async def lifespan(app):
    print("Loading models…")
    for name in MODEL_CONFIGS:
        try_load(name)
    if not loaded:
        raise RuntimeError("No models loaded — check weights paths")
    print(f"Ready: {list(loaded.keys())}\n")
    yield


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def gradcam(model: nn.Module, target_layer: nn.Module,
            x: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Grad-CAM using tensor-level gradient hooks — reliable even when all
    model parameters have requires_grad=False (inference-only loading).
    """
    activations: list = [None]
    gradients:   list = [None]

    h_fwd = target_layer.register_forward_hook(
        lambda m, i, o: activations.__setitem__(0, o)
    )

    with torch.enable_grad():
        out = model(x.detach().requires_grad_(True))
        act = activations[0]
        h_bwd = act.register_hook(lambda g: gradients.__setitem__(0, g))
        model.zero_grad()
        out[0, class_idx].backward()

    h_fwd.remove()
    h_bwd.remove()

    act_d = act.detach()
    grd   = gradients[0]

    if grd is None:
        weights = act_d.mean(dim=(2, 3), keepdim=True)
    else:
        weights = grd.mean(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * act_d).sum(1)).squeeze().cpu().numpy()
    if cam.ndim == 0:
        cam = np.array([[float(cam)]])
    vmin, vmax = cam.min(), cam.max()
    if vmax > vmin:
        cam = (cam - vmin) / (vmax - vmin)
    return cam


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    W, H  = img.size
    cam_u = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS))
    heat  = Image.fromarray((plt.get_cmap("jet")(cam_u / 255.0)[:, :, :3] * 255).astype(np.uint8))
    blend = Image.blend(img.convert("RGB"), heat, alpha)
    buf   = io.BytesIO()
    blend.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="NeuroScan API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models": list(loaded.keys())}


def _infer(entry: dict, img: Image.Image) -> tuple:
    """Run inference on a single loaded model entry. Returns (probs, tensor, img_resized)."""
    size   = entry["img_size"]
    tf     = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(entry["model"](tensor), dim=1)[0]
    return probs, tensor


@app.post("/predict")
async def predict(
    file:  UploadFile = File(...),
    model: str        = Query(default="efficientnet_b3"),
):
    if not loaded:
        raise HTTPException(503, "No models loaded")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    # ── Ensemble: average softmax of all loaded models ─────────────────────────
    if model == "ensemble":
        all_probs = []
        for entry in loaded.values():
            p, _ = _infer(entry, img)
            all_probs.append(p)
        probs = torch.stack(all_probs).mean(0)

        # Grad-CAM from EfficientNetB3 (or first available)
        cam_model = "efficientnet_b3" if "efficientnet_b3" in loaded else next(iter(loaded))
        cam_entry = loaded[cam_model]
        _, cam_tensor = _infer(cam_entry, img)
        display_name = "Ensemble"

    # ── Single model ───────────────────────────────────────────────────────────
    else:
        if model not in loaded:
            model = next(iter(loaded))
        cam_entry    = loaded[model]
        probs, cam_tensor = _infer(cam_entry, img)
        display_name = cam_entry["display"]

    class_idx  = int(probs.argmax())
    confidence = float(probs[class_idx])
    scores     = {c: round(float(probs[i]), 4) for i, c in enumerate(CLASS_NAMES)}

    heatmap: Optional[str] = None
    try:
        cam     = gradcam(cam_entry["model"], cam_entry["target_layer"], cam_tensor, class_idx)
        heatmap = overlay_heatmap(img, cam)
    except Exception as e:
        print(f"Grad-CAM error: {type(e).__name__}: {e}")

    return {
        "class":      CLASS_NAMES[class_idx],
        "confidence": confidence,
        "scores":     scores,
        "heatmap":    heatmap,
        "model":      display_name,
    }
