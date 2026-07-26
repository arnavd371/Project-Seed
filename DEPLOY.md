# Deploy Seed

## Live demo (Vercel)

**Production URL:** https://project-seed-phi.vercel.app

Vercel cannot run Streamlit or in-process XGBoost (bundle limit). The hosted demo is a **static site** under `public/` that serves **precomputed** model outputs for all 168 observed genotypes (`public/data/seed_payload.json`).

Redeploy:
```bash
npx vercel --prod
```

Regenerate payload after retraining:
```bash
# from repo root, with ML deps installed
python -c "..."  # or re-run the score export script used in deploy
```

## Streamlit (full interactive UI)

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py --server.port 8502
```

Streamlit Community Cloud:
- Repo: `arnavd371/Project-Seed`
- Main file: `streamlit_app.py`
- Python 3.11 (`runtime.txt`) + `packages.txt` (`libgomp1`)

## FastAPI (optional local)

`app/main.py` wraps `SeedScorer` for local API/HTML use. Not used on Vercel production (too large).

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
