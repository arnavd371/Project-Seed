# Seed - 3-Minute Demo Video Script

---

## Opening (0:00–0:20)

> "This is Seed - बीज - the first pharmacogenomics tool calibrated to Indian genomic data. Standard tools use European genomes. For 1.4 billion South Asians, the same gene and genotype can mean different prescribing risk. Seed re-calibrates CPIC using real 1000 Genomes South Asian data - fully offline, no LLM, no API keys."

**Screen:** Landing page → click **Enter Seed** → show stat grid (6,035 rows, 168 groups, 0 synthetic).

---

## Scenario 1: CYP2C19 *1/*2 - Clopidogrel (0:20–1:20)

**Tab:** Analyse

1. Select **CYP2C19** → ***1/*2** → **Analyse Genotype**
2. Point out **urgency banner** (Moderate) and **Evidence trail** (SAS freq vs EUR freq, SAS/EUR ratio)
3. Highlight **Dual-Model Comparison** cards:
   - Population-Only ML (~58% CV) - frequency structure only, no CPIC
   - Calibration ML (~96% CV) - adds CPIC clinical significance
4. Expand **Why this urgency?** - show SHAP/feature contribution bars
5. Show **gnomAD SAS vs NFE** chart - independent validation of *2 allele enrichment in South Asians
6. Read **Known limitations** box (168 groups, decision support only, not FDA device, VKORC1 excluded)

**Talking point:** "A cardiac patient on clopidogrel with *1/*2 is an intermediate metabolizer - more common in South Asia because *2 is enriched. Seed surfaces that population shift."

---

## Scenario 2: DPYD Poor Metabolizer - URGENT (1:20–2:00)

**Tab:** Demo Walkthrough → select **DPYD Poor Metabolizer**

1. Show scripted steps and **live preview** with URGENT banner
2. Contrast SAS vs EUR frequency for *2A/*2A
3. Emphasize: CPIC classifies this as **Urgent** - standard fluoropyrimidine doses can be life-threatening

**Talking point:** "Chemotherapy dosing is where population calibration matters most. DPYD poor metabolizers need urgent dose reduction - Seed flags this from real SAS observations."

---

## Simulations & SAS vs EUR (2:00–2:30)

**Tab:** Simulations

1. Select **CYP2C19** → play **animated bootstrap** (SAS vs EUR frequency shift)
2. Show **gene-level heatmap** across *1/*1, *1/*2, *2/*2

**Talking point:** "Monte Carlo resampling shows how allele-frequency uncertainty propagates - the same CPIC phenotype can imply different population prevalence."

---

## Model Performance & Honesty (2:30–2:50)

**Tab:** Model Performance

1. Side-by-side table: Population-Only ~58% vs Calibration ~96% vs Global Baseline
2. **Overfitting diagnosis** - train vs grouped-CV gap (threshold 0.10)
3. **Gene-held-out CV** - harder generalization test
4. **LOSO** subpopulation validation

**Talking point:** "We report honest grouped CV metrics - each genotype counts once, not once per person. The population-only model proves frequency structure alone is hard; calibration with CPIC context reaches ~96%."

---

## Close (2:50–3:00)

> "Seed is open source, runs offline on any laptop, supports Hindi summaries, and is ready for Streamlit Cloud deploy. Decision support only - always consult a clinician. GitHub link in the sidebar."

**Screen:** Impact tab → SDG mapping → Technical tab → tech stack.

---

## Pre-recording checklist

- [ ] Run `python model/compare_models.py` so `comparison_results.json` is current
- [ ] Confirm `india_population_only_xgb.pkl` exists in `model_artifacts/`
- [ ] Test CYP2C19 *1/*2 and DPYD `c.1905+1G>A (*2A)/c.1905+1G>A (*2A)` score without errors
- [ ] Enable large-text mode once to show accessibility
- [ ] Switch to Hindi briefly on Analyse tab
