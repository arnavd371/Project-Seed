# Deploy Seed on Streamlit Cloud

Seed is a Streamlit app (persistent Python process). It cannot run on Vercel.
Use Streamlit Community Cloud for the public demo URL.

## Prerequisites

- Public GitHub repo: https://github.com/arnavd371/Project-Seed
- Model artifacts under `model/model_artifacts/`
- Training data under `data/`
- `packages.txt` includes `libgomp1` (required by XGBoost)

## Streamlit Cloud setup

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** / **New app**.
3. Set:
   - **Repository:** `arnavd371/Project-Seed`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings**:
   - **Python version:** 3.11 (also set in `runtime.txt`)
   - **Secrets:** none required (fully offline app)
5. Deploy.

Expected URL form: `https://<app-name>.streamlit.app`

Streamlit Cloud runs `pip install -r requirements.txt` from the repo root.

## Required files in repo

```
├── app.py                          # Main Streamlit entry
├── requirements.txt
├── packages.txt                    # libgomp1 for XGBoost
├── runtime.txt                     # python-3.11
├── .streamlit/config.toml
├── data/
│   ├── real_training_data.json
│   ├── gene_drug_map.json
│   └── gnomad_sas_reference.json
├── model/
│   ├── seed_scorer.py
│   ├── phenotype_classifier.py
│   └── model_artifacts/
│       ├── india_calibrated_xgb.pkl
│       ├── india_population_only_xgb.pkl
│       ├── baseline_xgb.pkl
│       └── india_risk_regressor.pkl
└── utils/
```

## Local test before deploy

```bash
cd "project seed"   # or clone Project-Seed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open http://localhost:8502 → **Enter Seed** → walk **Analyse**, **Simulations**, **Demo Walkthrough**, **Model Performance**.

## Rebuild artifacts (optional)

If you change training code, regenerate artifacts locally and commit:

```bash
cd model
source ../.venv/bin/activate
python compare_models.py
python generate_plots.py
```

**Do not** commit `.venv/` or secrets.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError` for `.pkl` | Commit `model/model_artifacts/` |
| Import error for `phenotype_classifier` | Keep `model/` folder intact; app adds it to `sys.path` |
| SHAP slow on Cloud | App falls back to feature-importance proxy automatically |
| Wide layout not applied | Confirm `.streamlit/config.toml` is in repo root |
| XGBoost import fails on Cloud | Confirm `packages.txt` has `libgomp1` |
