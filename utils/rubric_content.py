CITATIONS = [
    {
        "id": 1,
        "text": "India represents ~17% of world population but <3% of genomic study participants.",
        "source": "Sirugo G, Williams SM, Tishkoff SA. The Missing Diversity in Human Genetic Studies. Cell 2019;177(1):26-31.",
    },
    {
        "id": 2,
        "text": "CYP2C19 poor-metaboliser alleles show substantially different frequencies across South Asian vs European cohorts.",
        "source": "CPIC Guideline for Clopidogrel and CYP2C19 (cpicpgx.org)",
    },
    {
        "id": 3,
        "text": "DPYD deficiency causes severe, sometimes fatal fluoropyrimidine toxicity; pre-emptive testing is CPIC Level 1A.",
        "source": "CPIC Guideline for Fluoropyrimidines and DPYD (cpicpgx.org)",
    },
    {
        "id": 4,
        "text": "Real pharmacogenomic training data parsed from 1000 Genomes Project South Asian superpopulation.",
        "source": "Sherman RM, Claw KG, Lee SB. Sci Rep 14, 22774 (2024).",
    },
    {
        "id": 5,
        "text": "NUDT15 variants strongly associated with thiopurine toxicity; frequencies differ in South Asian populations.",
        "source": "CPIC Guideline for Thiopurines and NUDT15/TPMT (cpicpgx.org)",
    },
]

SDG_MAPPING = [
    {
        "sdg": "SDG 3 - Good Health and Well-being",
        "mapping": "Safer prescribing for Indian patients by calibrating pharmacogenomic risk to South Asian allele frequencies (CYP2C19/clopidogrel, DPYD/5-FU, NUDT15/thiopurines).",
    },
    {
        "sdg": "SDG 10 - Reduced Inequalities",
        "mapping": "Addresses genomic under-representation: tools trained on European data systematically mis-estimate risk for 1.4B South Asians.",
    },
    {
        "sdg": "SDG 9 - Industry, Innovation and Infrastructure",
        "mapping": "Offline, open-source precision-medicine infrastructure deployable in low-bandwidth clinics without cloud dependency.",
    },
    {
        "sdg": "SDG 12 - Responsible Consumption",
        "mapping": "No cloud LLM inference - runs on a laptop, zero recurring API cost, minimal compute carbon vs. always-on cloud AI.",
    },
]

TARGET_AUDIENCE = [
    ("Primary", "Clinicians, pharmacists, and genetic counsellors serving South Asian patients in India and the diaspora."),
    ("Secondary", "Public-health researchers, medical students, and hospital IT teams evaluating offline PGx decision support."),
    ("Affected populations", "Patients from GIH (Gujarati), PJL (Punjabi), ITU (Tamil), STU (Telugu), and BEB (Bengali) subpopulations represented in training data."),
]

WHY_AI_NOT_RULES = """
CPIC lookup alone gives **clinical_significance** from phenotype text - Seed's rule baseline
achieves ~100% on that task. The AI layer adds what rules cannot:

1. **Predict population_adjusted_significance from frequency structure alone** - an XGBoost
   classifier using 8 population features (no CPIC score) on grouped CV (~65% accuracy).
2. **Non-linear risk calibration** - XGBRegressor learns continuous India risk scores from
   population frequencies + clinical context (R²≈0.99 on grouped CV vs simpler heuristics).
3. **Three-way comparison** - CPIC rule baseline vs European-feature ML vs India population
   ML on identical 5,546 real SAS individuals, with bootstrap CIs and LOSO validation.

XGBoost is the **primary technology**: gradient boosting on real genomic population statistics,
not a spreadsheet with an AI sticker.
"""

TECH_STACK = {
    "languages": ["Python 3.13"],
    "ml": ["XGBoost", "scikit-learn", "imbalanced-learn (evaluated, SMOTE rejected)"],
    "data": ["pandas", "numpy", "1000 Genomes Project combined.csv"],
    "ui": ["Custom Seed web UI (PI-inspired)", "Streamlit app (streamlit_app.py)", "Plotly/Matplotlib in full app"],
    "reporting": ["fpdf2 (PDF)", "algorithmic Hindi templates (no LLM)"],
    "hardware": "General-purpose laptop/desktop (Apple Silicon / x86). No GPU required. No AI-optimised hardware dependency.",
    "deployment": "Public Vercel demo (project-seed-phi.vercel.app) + GitHub open source. Full offline Streamlit app in repo.",
}

GTM_STRATEGY = [
    ("Phase 1 - Now", "Open-source on GitHub. Free Streamlit demo for clinicians. Zero subscription cost."),
    ("Phase 2 - Clinics", "Package as offline `.zip` for rural hospitals: pre-built model artifacts + Streamlit, no internet needed after install."),
    ("Phase 3 - Integration", "Export API-compatible JSON reports for hospital EMR systems; partner with AIIMS/regional medical colleges for validation studies."),
    ("Sustainability", "Community-maintained open dataset pipeline; CPIC guideline updates re-run through `parse_real_data.py` without vendor lock-in."),
]

ETHICS_PRIVACY = [
    ("Ethical use", "Decision-support only - not a diagnostic device. Urgency banners flag life-threatening chemo-toxicity genotypes (DPYD/NUDT15/TPMT)."),
    ("Privacy", "All inference runs locally. No patient data uploaded. No telemetry. No API keys. Training uses anonymised public 1000 Genomes IDs only."),
    ("Bias mitigation", "Grouped CV on real SAS individuals only; no SMOTE or synthetic rows; European baseline included for comparison."),
    ("Environmental", "No cloud LLM calls. XGBoost inference on CPU uses milliwatts vs. continuous GPU cloud hosting. Offline = low-bandwidth + low carbon."),
]

GROUPED_CV_EXPLANATION = """
**What is grouped cross-validation?**

Our dataset has **6,035 rows** (one per sequenced SAS person) but only **168 unique gene|genotype**
combinations. Every feature is the same for all people who share the same genotype.

**Plain k-fold CV is wrong here:** it can put the same genotype in both train and test, so the
model "cheats" by memorizing repeated rows → fake ~100% accuracy.

**Grouped CV fixes this:** we group rows by `gene|genotype`. All rows for one genotype stay
together in either train **or** test - never both. The model is tested on genotypes it has
never seen during that fold's training.

**Metrics are reported at group level:** each unique genotype counts **once**, not 157 times.
This is the honest measure of "can the model predict a new genotype it hasn't seen?"
"""

ACCESSIBILITY_FEATURES = [
    "English + Hindi clinical summaries and PDF reports",
    "High-contrast PI-style boxes (black border, white card) for readability",
    "Large-text mode toggle in sidebar",
    "Offline operation - no broadband required after setup",
    "Free and open-source - no paywall for target audience",
    "Dropdown genotype selection (no free-text parsing errors)",
]
