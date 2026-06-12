# NeuroScan — Implementation Plan

## Overview

Two deployments:
- **Frontend** — React/Vite static build → GitHub Pages (`TheAce10.github.io/NeuroScan`)
- **Backend** — FastAPI + PyTorch → Hugging Face Spaces (`The-Ace-000/neuroscan-api`)
- **Weights** — model `.pt` files → Hugging Face Hub (`The-Ace-000/neuroscan-weights`)

---

## Phase 1 — Hugging Face Hub (Weights)

**Goal:** Host the three model weight files publicly so the backend can download them on startup.

Steps:
1. Create model repo `The-Ace-000/neuroscan-weights` on HF Hub
2. Upload the three files:
   - `efficientnet_b3_neuroscan.pt`
   - `vgg16_phase2_best.pt`
   - `densenet121_neuroscan.pt`
3. Note the download URLs (format: `https://huggingface.co/The-Ace-000/neuroscan-weights/resolve/main/<filename>`)

Status: [ ] TODO

---

## Phase 2 — Hugging Face Spaces (Backend)

**Goal:** Run the FastAPI backend as a public API endpoint.

HF Spaces uses a `app.py` entry point. The backend needs to:
1. Download weights from HF Hub on first startup (cached after)
2. Expose the same `/health` and `/predict` endpoints

Files needed in the Space:
- `app.py` — modified `main.py` that downloads weights via `huggingface_hub`
- `requirements.txt` — Python dependencies

Space URL will be: `https://the-ace-000-neuroscan-api.hf.space`

Status: [ ] TODO — depends on Phase 1

---

## Phase 3 — GitHub Pages (Frontend)

**Goal:** Serve the Vite build at `TheAce10.github.io/NeuroScan`.

Changes needed:
1. `src/web/neuroscan-web/vite.config.js` — add `base: '/NeuroScan/'`
2. `src/web/neuroscan-web/src/App.jsx` — update `DEFAULT_ENDPOINT` to the HF Spaces URL
3. `.github/workflows/deploy.yml` — GitHub Actions: build Vite on push to `main`, deploy to `gh-pages` branch
4. Enable GitHub Pages in repo Settings → Pages → Source: `gh-pages` branch

Status: [ ] TODO — depends on Phase 2 (needs the live backend URL)

---

## Phase 4 — Wire Up & Test

1. Set `DEFAULT_ENDPOINT` in `App.jsx` to `https://the-ace-000-neuroscan-api.hf.space`
2. Push to `main` → GitHub Actions builds and deploys automatically
3. Visit `https://TheAce10.github.io/NeuroScan`
4. Upload a test MRI scan, confirm predictions and heatmap load

---

## Phase 5 — Fill in Scaffolded Folders

- `results/` — save classification reports from `eval_models.py` as `.txt` / `.csv`
- `figures/` — confusion matrix plots per model + comparison bar chart
- `tests/` — pytest tests for `/health` and `/predict` endpoints
- `docs/` — API reference (endpoints, request/response shapes)

Status: [ ] TODO

---

## File Change Summary

| File | Change | Phase |
|---|---|---|
| `src/web/neuroscan-web/vite.config.js` | Add `base: '/NeuroScan/'` | 3 |
| `src/web/neuroscan-web/src/App.jsx` | Update `DEFAULT_ENDPOINT` | 3 |
| `src/web/neuroscan-web/backend/main.py` | Add HF Hub weight download on startup | 2 |
| `.github/workflows/deploy.yml` | New — GitHub Actions CI/CD | 3 |
| `data/README.md` | Add HF Hub model repo link | 1 |
| `results/classification_report.txt` | New — eval output | 5 |
| `figures/confusion_matrix_*.png` | New — per-model confusion matrices | 5 |
| `tests/test_api.py` | New — smoke tests | 5 |

---

## Dependencies / Blockers

- Phase 2 requires HF account token (set as GitHub secret `HF_TOKEN` for Actions)
- Phase 3 requires Phase 2 live URL to hardcode the default endpoint
- GitHub Pages must be enabled manually in repo Settings after first Actions run
