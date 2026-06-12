# Contributing to NeuroScan

## Branch strategy
- `main` — stable releases only
- `dev` — integration branch; open PRs against this
- `feature/<name>` — feature branches off `dev`

## Setup
```bash
conda env create -f environment.yml
conda activate neuroscan
cd src/web/neuroscan-web && npm install
```

## Running locally
```bash
# Backend
cd src/web/neuroscan-web/backend
uvicorn main:app --reload

# Frontend
cd src/web/neuroscan-web
npm run dev
```

## Adding a model
1. Train on Kaggle using a notebook in `notebooks/`
2. Add builder + config entry in `src/web/neuroscan-web/backend/main.py`
3. Add option to `src/web/neuroscan-web/src/components/ModelSelector.jsx`
4. Upload weights to Hugging Face Hub
5. Update `data/README.md` with the new weight path

## Running tests
```bash
cd tests
pytest
```

## Code style
- Python: follow PEP 8
- JavaScript: ESLint config at `src/web/neuroscan-web/eslint.config.js`
