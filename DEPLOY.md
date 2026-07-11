# Deploy Seed on Streamlit Cloud

## Prerequisites

- GitHub repo with `seed/` at repository root (or adjust paths below)
- Model artifacts committed under `seed/model/model_artifacts/`
- Training data committed under `seed/data/`

## Streamlit Cloud setup

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Set:
   - **Repository:** `arnavd371/arnavdhiman` (or your fork)
   - **Branch:** `main`
   - **Main file path:** `seed/app.py`
4. Click **Advanced settings**:
   - **Python version:** 3.11 or 3.12
   - **Secrets:** none required (fully offline app)
5. Deploy.

Streamlit Cloud runs `pip install -r requirements.txt` from the directory containing `app.py`. Ensure `seed/requirements.txt` lists all dependencies.

## Required files in repo

```
seed/
├── app.py                          # Main Streamlit entry
├── requirements.txt
├── .streamlit/config.toml          # PI theme, wide layout
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
cd seed
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open http://localhost:8502 → **Enter Seed** → walk **Analyse**, **Simulations**, **Demo Walkthrough**, **Model Performance**.

## Rebuild artifacts (optional)

If you change training code, regenerate artifacts locally and commit:

```bash
cd seed/model
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
| Wide layout not applied | Confirm `.streamlit/config.toml` is in `seed/` |
