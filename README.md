# Seed | बीज
## India-Calibrated Pharmacogenomics

### What does "Seed" mean?
Seed (बीज, Beej) is the Hindi and Sanskrit word for seed
or genetic origin. Your genome is the seed from which your
health grows. Medicine should match your seed - not someone
else's. Seed is the first pharmacogenomics tool calibrated
to Indian genetic data.

Standard pharmacogenomic guidelines (CPIC, FDA labels) are calibrated almost
entirely on European and East Asian genomic cohorts. Seed is trained
**entirely on real 1000 Genomes Project South Asian (SAS) genomes** - zero
synthetic training rows - to show, with real numbers, where that gap
matters for ~1.4 billion South Asians.

Seed runs **completely offline**. There are no LLM calls, no API keys,
and no external network dependency at inference time. Two XGBoost
classifiers, a purely algorithmic PDF/Hindi-summary generator, and Plotly/
Matplotlib visualisations - all running on-device from local data files.

---

## The Core Idea

For a given gene + genotype, Seed reports:

1. **Observed real population frequencies** - South Asian (SAS), European
   (EUR), and East Asian (EAS) - computed directly from real 1000 Genomes
   individuals, not estimated.
2. **Two XGBoost model predictions** for the same genotype: a **Global
   Baseline** model (trained using only European-frequency features) and an
   **India-Calibrated** model (trained using South Asian frequency, the
   SAS-vs-EUR ratio, and gene-importance features) - both trained on the
   *exact same real SAS individuals*, so the only difference is which
   population signal each model is allowed to see.
3. A continuous **India Risk Score** (0.0–1.0) and its delta from the
   baseline score, so the population-specific contribution is visible as a
   number, not just a category.

### Why this matters

- `CYP2C19*2` (clopidogrel non-response, relevant to post-stent cardiac
  care) is measurably more common in this dataset's South Asian sample
  (35.8%) than its European sample (14.5%) - see the CYP2C19 population
  chart in the Model Performance tab, computed live from the real data.
- `DPYD`, `NUDT15`, and `TPMT` poor-metaboliser genotypes carry
  life-threatening chemotherapy toxicity risk (severe neutropenia,
  mucositis) - pre-emptive testing guidance calibrated on the wrong
  population frequency estimates real-world risk incorrectly for
  under-represented populations.
- Precision-medicine infrastructure that only reflects European allele
  frequencies is precision medicine that under-serves the majority of the
  world's population.

---

## Honesty About Methodology (please read before citing any number here)

This project went through **two rounds of methodological correction**
during development, and we consider disclosing both - rather than hiding
them - essential to this project's scientific credibility.

### 1. Grouped cross-validation (fixes a serious data-leakage bug)

`data/real_training_data.json` has **5,546 rows, one per sequenced SAS
individual**, but every model feature (diplotype frequency, SAS/EUR ratio,
gene importance, etc.) is a **(gene, genotype)-level statistic** - it does
not vary per individual. There are only **166 unique (gene, genotype)
combinations** in the whole dataset, so the average genotype is duplicated
**~33x** (e.g. `CYP2B6 *1/*6` alone is 157 identical rows).

An initial implementation used plain row-level `StratifiedKFold` for cross-
validation. This put duplicate rows for the same genotype in **both the
train and test fold**, so "cross-validated" accuracy was mostly measuring
memorization of repeated rows, not generalization to genotypes the model
hasn't seen. This produced a **not-credible ~100% accuracy, macro F1, and
Cohen's Kappa across every class - including a class with only 4 samples**.
That result was caught, diagnosed, and fixed before this model was shipped.

**The fix:** both `model/train_baseline.py` and `model/train_model.py` now
use **`StratifiedGroupKFold`**, grouped by `gene|genotype`
(`model/phenotype_classifier.py`'s `build_group_keys()` /
`make_cv_splitter()`), so every genotype's rows land in exactly one fold -
no genotype is ever memorized from its own duplicate in the test set.

The **"Urgent" class (class 3)** has only **2 unique genotype groups**
project-wide (1 DPYD homozygote, `c.1905+1G>A (*2A)/c.1905+1G>A (*2A)`,
n=1 individual; 1 NUDT15 `*3/*3` group, n=3 individuals). With a grouped
splitter this means at most 2 of the 5 CV folds ever hold out a class-3
example - the reported class-3 precision/recall/F1 (typically ~0.4 F1,
0.25 recall in our runs) is a genuine, non-leaked, held-out-genotype
result, but it is **based on only 2 independent genotype groups** and
should be read as **directional, not statistically powerful**. This is
disclosed in the training scripts' printed output, in
`model/compare_models.py`'s output, and in the app's sidebar / Model
Performance tab - nowhere is it silently rounded up to "high confidence."

### 2. Label/feature circularity (a known, disclosed limitation)

`clinical_significance` (the label both models predict) and
`cpic_phenotype_score` (a feature both models share) are both deterministic
functions of the **same underlying CPIC `Phenotype` string** in the source
data (see `data/parse_real_data.py`'s `PHENOTYPE_MAP` / `CPIC_SCORE_MAP`).
That means this specific 0–3 classification task is largely solvable from
CPIC score plus gene identity alone, **with or without any population
frequency information** - so the accuracy delta between the Global
Baseline and India-Calibrated models is small (in our runs, ~0.00) and
**this is expected, not a sign that India-calibration has no value.** It
means this particular classification label is not where that value shows
up. Seed's population-specific contribution is visible in the
SAS-vs-European frequency ratio and the continuous India Risk Score shown
in the **Analyse** tab - e.g. a genotype that is 30x more common in South
Asians than Europeans shows exactly that ratio and a correspondingly
adjusted score, independent of whichever CPIC category it falls into.

We report the real measured numbers either way, honestly, in
`model/model_artifacts/comparison_table.txt` and throughout the app.

---

## Architecture

```
data/parse_real_data.py
  1000 Genomes Project combined.csv (real genotypes, all populations)
      │  filter: SAS individuals, CPIC-actionable genes
      │  compute real observed SAS + EUR diplotype frequencies
      ▼
  data/real_training_data.json   (5,546 real-individual rows)
      │
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
model/train_baseline.py       model/train_model.py         model/generate_plots.py
  4 features (EUR-only)         9 features (+ SAS freq,       learning curve,
  StratifiedGroupKFold          SAS/EUR ratio, etc.)          feature importances,
  by gene|genotype              StratifiedGroupKFold          confusion matrix,
      │                         by gene|genotype               CYP2C19 population chart
      ▼                              ▼
  baseline_xgb.pkl              india_calibrated_xgb.pkl
                                 feature_engineer.pkl
      │                              │
      └──────────────┬───────────────┘
                      ▼
           model/seed_scorer.py
           (looks up a gene+genotype's real data,
            runs both models, computes India Risk Score)
                      │
                      ▼
                   app.py (Streamlit)
       ┌─────────────┬─────────────┬─────────────┐
       │  Analyse    │   Model     │   About &   │
       │  tab        │ Performance │   Impact    │
       └─────────────┴─────────────┴─────────────┘
```

---

## Project Structure

```
seed/
  app.py                          Streamlit UI (dark navy / teal, fully offline)
  data/
    parse_real_data.py            Parses 1000 Genomes combined.csv -> real_training_data.json
    real_training_data.json       5,546 real SAS-individual rows (generated)
    gene_drug_map.json             Gene -> drugs/CPIC-level/rsIDs (12 usable genes)
  model/
    phenotype_classifier.py       Feature schemas + grouped-CV helpers (shared)
    train_baseline.py             Trains the Global Baseline model (EUR-only features)
    train_model.py                Trains the India-Calibrated model (+ SAS features)
    compare_models.py             Runs both, prints/saves the comparison table
    generate_plots.py             Learning curve, feature importances, confusion
                                   matrix, CYP2C19 population chart
    seed_scorer.py           Looks up real data + runs both models for the app
    model_artifacts/               baseline_xgb.pkl, india_calibrated_xgb.pkl,
                                    feature_engineer.pkl, comparison_results.json,
                                    comparison_table.txt, *.png plots
  utils/
    hindi_templates.py            Offline, hardcoded Hindi summary templates
    pdf_generator.py               fpdf2-based PDF clinical summary generator
    star_allele_caller.py          Simplified demo rsID -> star-allele caller (VCF path)
    vcf_parser.py                  Minimal, dependency-free VCF reader
    visualizer.py                  Plotly population-comparison chart
  requirements.txt
  README.md
```

---

## Data Sources

- **1000 Genomes Project**, phase 3 (`combined.csv`, 2,504 individuals,
  145,232 gene x sample rows), filtered to the **South Asian (SAS)**
  superpopulation (**GIH, PJL, ITU, STU, BEB** - 489 real individuals) for
  training, and to **EUR** / **EAS** for the comparison frequencies shown in
  the app.
- **CPIC** - Clinical Pharmacogenetics Implementation Consortium phenotype
  and gene-drug guideline mappings. <https://cpicpgx.org/>
- **PharmVar** - allele/haplotype definitions (used only for rsIDs in the
  simplified demo VCF star-allele caller, not the primary real-data
  pipeline). <https://www.pharmvar.org/>

### Known data limitations

- **VKORC1 is excluded** from the app's selectable gene list: every SAS
  `VKORC1` phenotype call in this dataset is `Indeterminate`, so it has zero
  usable training rows.
- **12 genes are usable**: ABCG2, CYP2B6, CYP2C19, CYP2C9, CYP2D6, CYP3A5,
  DPYD, IFNL3, NUDT15, SLCO1B1, TPMT, UGT1A1.
- A small fraction (~25%) of (gene, genotype) groups have a European
  reference frequency of exactly 0 in this dataset (the genotype simply
  wasn't observed in the EUR sample here). The app reports this explicitly
  as "not observed in the European reference sample" rather than a
  misleading, division-by-near-zero ratio like "620000x more common."
- See **Methodology** above for the grouped-CV and label-circularity
  disclosures.

---

## Setup

```bash
cd seed
python3 -m venv .venv
source .venv/bin/activate        # on macOS/Linux
pip install -r requirements.txt
```

Seed expects the raw 1000 Genomes CSV at
`~/Downloads/1kgp-pgx-data/data/combined.csv` (used by
`data/parse_real_data.py` to regenerate `real_training_data.json`, and by
the app for the live East-Asian comparison frequency). This file is not
committed to the repository; the parsed `real_training_data.json` is.

## Rebuilding the Models From Scratch

```bash
cd seed
source .venv/bin/activate

python data/parse_real_data.py          # combined.csv -> real_training_data.json
python model/train_baseline.py          # -> model_artifacts/baseline_xgb.pkl
python model/train_model.py             # -> model_artifacts/india_calibrated_xgb.pkl
python model/compare_models.py          # -> comparison_results.json / comparison_table.txt
python model/generate_plots.py          # -> 4 PNG plots in model_artifacts/
```

Each script prints its methodology notes (grouped-CV fold counts, SMOTE
decision, class distributions) to stdout as it runs.

## Running the App

```bash
cd seed
source .venv/bin/activate
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
Since there is no LLM and no API key, the app is ready to use immediately
after training the models once.

### Using the app

1. Click **Enter Seed** on the landing page.
2. On the **Analyse** tab, pick a gene and a genotype (only genotypes
   actually observed in the real SAS dataset are offered, with their real
   sample counts) and click **Analyse Genotype**.
3. Review the urgency banner, India Risk Score vs. Global Baseline score,
   population frequency comparison (SAS/EUR/EAS), SAS subpopulation
   breakdown, affected medications, and Hindi summary. Download a PDF
   clinical summary if needed.
4. The **Model Performance** tab shows the full comparison table, the
   methodology caveats, and the four generated plots.
5. The **About & Impact** tab has the full problem statement, architecture,
   and impact information.

---

## Suggested Test Cases

| Gene | Genotype | Real n | Expected result |
|---|---|---|---|
| `NUDT15` | `*3/*3` | 3 | **Urgent** - thiopurine (azathioprine, mercaptopurine, thioguanine) poor-metaboliser toxicity risk |
| `DPYD` | `c.1905+1G>A (*2A)/c.1905+1G>A (*2A)` | 1 | **Urgent** - fluoropyrimidine (5-FU, capecitabine) poor-metaboliser toxicity risk |
| `CYP2C19` | `*2/*2` | 74 | **Significant** - clopidogrel poor-metaboliser, high real sample count |
| `CYP2C19` | `*1/*17` | 73 | **Moderate** - increased-function/ultrarapid, common in this SAS sample |
| `ABCG2` | `Reference/Reference` | 398 | **No Action** - most common genotype in the dataset |

---

## Ethical Disclaimer

Seed is a **decision-support research prototype**, not a diagnostic
device. It is not FDA/CDSCO-cleared and has not been clinically validated
in Indian hospitals. It must not be used to make or change any prescribing
decision without review by a qualified healthcare professional.

The "Urgent" clinical-significance class is supported by only 2 unique real
genotype groups in this dataset (see *Methodology* above) - its metrics
should be read as directional evidence that the modelling approach works,
not as a validated estimate of real-world sensitivity for rare, severe
genotypes. A production clinical tool would need substantially more Urgent-
class data before its recall/precision on that class could be trusted for
patient care.

Pharmacogenomic results must always be interpreted alongside full clinical
context, co-medications, organ function, and other patient-specific
factors.

---

## Citation

```
Seed: An India-Calibrated XGBoost Pharmacogenomic Variant
Interpreter, trained on real 1000 Genomes Project South Asian genomes. 2026.

Underlying data: The 1000 Genomes Project Consortium, "A global reference
for human genetic variation," Nature 526, 68-74 (2015); CPIC
(cpicpgx.org) gene-drug and phenotype guidelines.
```
