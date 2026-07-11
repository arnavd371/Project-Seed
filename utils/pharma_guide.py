MANIFESTO = """
Seed exists because pharmacogenomic guidelines were built mostly on European and East Asian
cohorts. For South Asian patients, the same star allele can be far more or less common. That changes
how often a "moderate" CPIC label shows up in daily practice, even when the biology is unchanged.

We do not replace CPIC. We show when population frequency should change how urgently you act on a
CPIC phenotype. Every frequency comes from real 1000 Genomes South Asian genomes. No synthetic
training rows. No cloud LLM. Decision support only: always confirm with a qualified prescriber.
"""

QUICK_START = [
    (
        "Step 1: Pick a gene and genotype",
        "Open the Analyse tab. Genes and genotypes are limited to what was actually observed in "
        "489 South Asian reference genomes (GIH, PJL, ITU, STU, BEB). If a genotype is missing, "
        "Seed cannot score it honestly.",
    ),
    (
        "Step 2: Read the urgency banner",
        "Green to black banner shows CPIC-based clinical significance for this genotype. "
        "Urgent (black) means standard doses of linked drugs may be unsafe (for example DPYD with "
        "fluoropyrimidines).",
    ),
    (
        "Step 3: Compare the two AI models",
        "Population-Only ML (~59% accuracy) uses allele frequencies only. It proves frequency "
        "structure alone is a hard problem. Calibration ML (~96% accuracy) adds CPIC clinical "
        "context plus population adjustment. Users can see both 'AI is necessary' and 'AI adds "
        "value on top of CPIC.'",
    ),
    (
        "Step 4: Check SAS vs European frequency",
        "Bar charts and SAS/EUR ratio explain why South Asian prevalence matters. Example: CYP2C19 "
        "*2 is enriched in South Asians, so clopidogrel intermediate metabolizer genotypes are "
        "more common than European tools assume.",
    ),
    (
        "Step 5: Open 'Why this urgency?'",
        "Feature contribution bars (SHAP when available) show which inputs pushed the model toward "
        "Moderate, Significant, or Urgent.",
    ),
    (
        "Step 6: Read limitations before acting",
        "168 unique genotype groups, VKORC1 excluded, not an FDA-cleared device. Use for research "
        "and education, not as a sole prescribing authority.",
    ),
]

GLOSSARY = [
    ("CPIC", "Clinical Pharmacogenetics Implementation Consortium. Publishes gene/drug dosing guidance."),
    ("SAS", "South Asian superpopulation in 1000 Genomes (Indian and Bangladeshi reference samples)."),
    ("Star allele", "Named pharmacogene variant, for example CYP2C19 *2."),
    ("Diplotype", "Pair of star alleles, for example *1/*2."),
    ("Population-adjusted significance", "Urgency class after weighing CPIC label with SAS vs EUR frequency."),
    ("Grouped CV", "Cross-validation that keeps all rows for one genotype in the same fold (prevents fake 100% accuracy)."),
]

DEMO_SCENARIOS = [
    {
        "name": "CYP2C19 *1/*2 + clopidogrel",
        "gene": "CYP2C19",
        "genotype": "*1/*2",
        "why": "Intermediate metabolizer. *2 is more common in South Asians. Relevant for antiplatelet therapy after stenting.",
    },
    {
        "name": "DPYD poor metabolizer + fluoropyrimidine",
        "gene": "DPYD",
        "genotype": "c.1905+1G>A (*2A)/c.1905+1G>A (*2A)",
        "why": "Urgent toxicity risk with 5-FU or capecitabine. Shows life-threatening chemo flag.",
    },
]
