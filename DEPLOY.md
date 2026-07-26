# Deploy Seed

## Vercel (public web demo)

Seed's interactive web UI is a FastAPI app at `app/main.py` (Vercel cannot run Streamlit).

1. From this repo root:
   ```bash
   npx vercel --prod
   ```
2. Framework: FastAPI (`app/main.py`), Python 3.11+
3. No secrets required

Local FastAPI check:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Streamlit (full local / Cloud UI)

The original Streamlit app is `streamlit_app.py`.

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py --server.port 8502
```

Streamlit Community Cloud:
- Repository: `arnavd371/Project-Seed`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: 3.11 (`runtime.txt`)
- `packages.txt` includes `libgomp1` for XGBoost

## Required files

```
├── app/main.py                     # FastAPI (Vercel)
├── streamlit_app.py                # Full Streamlit UI
├── requirements.txt                # Vercel / FastAPI deps
├── requirements-streamlit.txt      # Full local Streamlit deps
├── packages.txt / runtime.txt      # Streamlit Cloud
├── data/
├── model/ (+ model_artifacts/)
└── utils/
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Vercel cold start slow / OOM | Models load on first request; maxDuration=60 in vercel.json |
| `FileNotFoundError` for `.pkl` | Commit `model/model_artifacts/` |
| XGBoost on Streamlit Cloud | Confirm `packages.txt` has `libgomp1` |
| Import error for `phenotype_classifier` | Keep `model/` on `sys.path` (set in `app/main.py`) |
