# Seed - Evaluation Rubric Mapping

**One-line pitch:** Seed proves that the same CPIC genotype carries different real-world prescribing risk in South Asians because allele frequencies differ - and shows it with honest ML (population-only ~59% vs calibrated ~96%) on real genomes only.

Each row maps a rubric criterion to evidence inside the app. Open the corresponding tab during evaluation.

## Metric 01: Impact & Inclusion (15 pts)

| Criterion | Evidence in Seed |
|-----------|----------------------|
| Problem clearly defined | **Impact & Inclusion tab** - problem box + 5 citations |
| Evidence of problem | Sirugo et al. 2019; CPIC guidelines; Sherman et al. 2024 |
| Target audience defined | Primary/secondary/affected subpopulations listed |
| Equivalent UX for all | Dropdown UI, large-text mode, high-contrast boxes |
| Accessibility | Large-text toggle, bilingual EN/HI, no color-only encoding |
| Financially viable | Free, open-source, runs on any laptop |
| Offline / low-bandwidth | Zero API calls; works after `pip install` + model artifacts |
| Multilingual | English UI + Hindi clinical summaries + PDF |
| Societal impact defined | Chemo toxicity, clopidogrel, 1.4B underserved |
| AI impact vs traditional software | XGBoost learns population-frequency interactions; rule tables cannot |
| SDG mapping | SDG 3, 10, 9, 12 table |
| Environmental implications | No cloud LLM; CPU-only inference |
| Sustainable pathways | Open-source GTM in AI Innovation tab |

## Metric 02: AI Innovation (20 pts)

| Criterion | Evidence in Seed |
|-----------|----------------------|
| Not a force-fit | AI Innovation tab - "Why AI" section |
| AI as primary element | Dual XGBoost models are core; not accessory |
| Originality | Population-only vs Calibration A/B on same real SAS data |
| AI knowledge demonstrated | Grouped CV, gene holdout, LOSO, SHAP explanations |
| Data obtain/analyse | Data pipeline section + Methods |
| Data justified | SMOTE rejected, grouped CV, real 1000G only |
| Ethics addressed | Ethics/privacy/bias boxes + limitations box on Analyse |
| Privacy | Local-only inference, no patient upload |
| Bias mitigation | Baseline comparison + gnomAD SAS validation + disclosed limitations |
| Environmental AI impact | Offline CPU inference |
| Working prototype | Analyse + Simulations + Demo Walkthrough tabs |
| Deployment | GitHub + Streamlit Cloud (`DEPLOY.md`) |
| GTM strategy | 3-phase table in AI Innovation tab |

## Metric 03: Technical Skills (15 pts)

| Criterion | Evidence in Seed |
|-----------|----------------------|
| Tech stack explained | Technical Skills tab |
| Hardware | General-purpose laptop (documented) |
| Software | Python, XGBoost, scikit-learn, Streamlit, Plotly, SHAP, fpdf2 |
| Custom UI | PI-style CSS (`utils/pi_theme.py`), built for this solution |
| AI packages | XGBoost + scikit-learn + SHAP (advanced classical ML) |

## Training methodology (FAQ)

### Real-only training

- **Source:** `data/real_training_data.json` - 6,035 rows from 489 SAS individuals (1000 Genomes)
- **Dedup:** `dedupe_records_by_group()` → 168 unique gene|genotype groups
- **Never used for training:** `genotype_catalog.json`, synthetic rows, SMOTE
- **Read-only references:** `frequency_reference.json`, `gnomad_sas_reference.json` (comparison only)

### Dual-model architecture

| Model | Features | Target | Grouped-CV accuracy |
|-------|----------|--------|---------------------|
| Global Baseline | 4 EUR-frequency features | clinical_significance | ~baseline |
| Population-Only | 19 population features, **no CPIC** | population_adjusted_significance | ~58% |
| Calibration ML | 19 population + CPIC clinical_significance | population_adjusted_significance | ~96% |

Population-only proves frequency structure alone is a hard task; calibration shows CPIC context + population features reach high accuracy without catalog leakage.

### Overfitting audit

- Train-set group-level accuracy vs grouped-CV accuracy
- Gap > 0.10 triggers stronger regularization (lower `max_depth`, higher `min_child_weight`)
- Reported in **Model Performance** tab and `comparison_results.json`

### Gene-held-out CV

- Leave-one-**gene**-out: hold all genotypes for one gene, train on remaining genes
- Harder than standard grouped CV - tests generalization to unseen genes
- Both Calibration and Population-Only models evaluated (`validate_gene_holdout.py`)

### LOSO validation

- Leave-one-SAS-subpopulation-out (GIH, PJL, ITU, STU, BEB)
- Group-level metrics - each genotype counts once

## Live demo

```bash
cd seed && source .venv/bin/activate && streamlit run app.py --server.port 8502
```

Open **http://localhost:8502** → Enter Seed → walk tabs:

1. **Analyse** - CYP2C19 *1/*2, dual models, SHAP, gnomAD, limitations
2. **Simulations** - Monte Carlo frequency animation
3. **Demo Walkthrough** - scripted CYP2C19 + DPYD scenarios
4. **Model Performance** - population-only vs calibration, gene holdout, overfitting

See `DEMO_SCRIPT.md` for a 3-minute video walkthrough.

## Streamlit Cloud deploy

See `DEPLOY.md` - main file `seed/app.py`, include `data/` and `model/model_artifacts/`.
