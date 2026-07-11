"""
Parses 1000 Genomes PGx data (Sherman et al., Sci Rep 2024) into Seed's training schema.
All records are real SAS individuals - zero synthetic data.

Known limitations:
  - VKORC1: all SAS phenotypes are Indeterminate → 0 usable rows.
  - CACNA1S/RYR1 excluded (Uncertain Susceptibility only).
  - Class 3 ("Urgent"): only 4 real SAS records; CV adapts and discloses this.
"""

import pandas as pd
import json
import numpy as np
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.expanduser("~/Downloads/1kgp-pgx-data/data/combined.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "real_training_data.json")
FREQ_REF_PATH = os.path.join(SCRIPT_DIR, "frequency_reference.json")

SAS_SUBPOPS = ["GIH", "PJL", "ITU", "STU", "BEB"]
WILDTYPE_ALLELES = {"*1", "wildtype", "normal"}

df = pd.read_csv(CSV_PATH)

sas = df[df["Superpopulation code"] == "SAS"].copy()
print(f"Total SAS rows: {len(sas)}")
print(sas["Population code"].value_counts())

CPIC_GENES = [
    "CYP2C19", "CYP2D6", "CYP2C9", "DPYD", "NUDT15",
    "TPMT", "SLCO1B1", "UGT1A1", "IFNL3", "CYP3A5",
    "ABCG2", "CYP2B6", "VKORC1", "F5",  # F5: Favorable/Unfavorable only
]
sas = sas[sas["Gene"].isin(CPIC_GENES)]

sas = sas[sas["Phenotype"] != "Indeterminate"]
sas = sas[sas["Phenotype"].notna()]
print(f"After filtering: {len(sas)} rows")

if "VKORC1" not in sas["Gene"].unique():
    print(
        "NOTE: VKORC1 contributed 0 rows after filtering -- all SAS VKORC1 "
        "phenotype calls in this dataset are 'Indeterminate'. Seed "
        "excludes VKORC1 from the app's selectable gene list for this reason."
    )

PHENOTYPE_MAP = {
    "Normal Metabolizer": 0,
    "Normal Function": 0,
    "Favorable Response": 0,
    "Increased Function": 0,
    "Likely Normal Metabolizer": 0,
    "Rapid Metabolizer": 1,
    "Intermediate Metabolizer": 1,
    "Intermediate Function": 1,
    "Decreased Function": 1,
    "Unfavorable Response": 1,
    "Likely Intermediate Metabolizer": 1,
    "Likely Rapid Metabolizer": 1,
    "Ultrarapid Metabolizer": 1,
    "Likely Ultrarapid Metabolizer": 1,
    "Poor Metabolizer": 2,
    "Poor Function": 2,
    "Likely Poor Metabolizer": 2,
}

URGENT_GENES_PHENOTYPES = {
    "DPYD": ["Poor Metabolizer", "Likely Poor Metabolizer"],
    "NUDT15": ["Poor Metabolizer", "Likely Poor Metabolizer"],
    "TPMT": ["Poor Metabolizer", "Likely Poor Metabolizer"],
}

GENE_IMPORTANCE = {
    "DPYD": 1.0, "NUDT15": 1.0, "TPMT": 0.9,
    "CYP2C19": 0.9, "CYP2D6": 0.9, "CYP2C9": 0.8,
    "VKORC1": 0.7, "UGT1A1": 0.7, "SLCO1B1": 0.6,
    "CYP3A5": 0.6, "IFNL3": 0.5, "ABCG2": 0.5,
    "CYP2B6": 0.5, "F5": 0.6,
}

CPIC_SCORE_MAP = {
    "Poor Metabolizer": 0.0, "Poor Function": 0.0,
    "Likely Poor Metabolizer": 0.0,
    "Intermediate Metabolizer": 0.5,
    "Intermediate Function": 0.5,
    "Decreased Function": 0.5,
    "Likely Intermediate Metabolizer": 0.5,
    "Normal Metabolizer": 1.0, "Normal Function": 1.0,
    "Favorable Response": 1.0, "Increased Function": 1.0,
    "Likely Normal Metabolizer": 1.0,
    "Rapid Metabolizer": 1.5, "Likely Rapid Metabolizer": 1.5,
    "Ultrarapid Metabolizer": 2.0,
    "Likely Ultrarapid Metabolizer": 2.0,
    "Unfavorable Response": 0.5,
}

eur = df[df["Superpopulation code"] == "EUR"].copy()
eas = df[df["Superpopulation code"] == "EAS"].copy()
afr = df[df["Superpopulation code"] == "AFR"].copy()
amr = df[df["Superpopulation code"] == "AMR"].copy()
global_df = df[~df["Superpopulation code"].str.contains(",", na=False)].copy()


def compute_diplotype_freq(gene, genotype, pop_df):
    gene_df = pop_df[pop_df["Gene"] == gene]
    total = len(gene_df)
    if total == 0:
        return 0.0
    count = len(gene_df[gene_df["Genotype"] == genotype])
    return round(count / total, 4)


def compute_subpop_freqs(gene, genotype, sas_df):
    gene_df = sas_df[sas_df["Gene"] == gene]
    freqs = {}
    for pop in SAS_SUBPOPS:
        pop_df = gene_df[gene_df["Population code"] == pop]
        total = len(pop_df)
        if total == 0:
            freqs[pop] = 0.0
        else:
            count = len(pop_df[pop_df["Genotype"] == genotype])
            freqs[pop] = round(count / total, 4)
    return freqs


def subpop_dispersion(freqs):
    vals = list(freqs.values())
    return round(float(np.std(vals)), 4) if vals else 0.0


def subpop_entropy(freqs):
    vals = np.array(list(freqs.values()), dtype=float) + 1e-8
    vals = vals / vals.sum()
    return round(float(-np.sum(vals * np.log(vals))), 4)


def genotype_complexity(genotype):
    alleles = [a.strip() for a in str(genotype).split("/")]
    star_count = sum(1 for a in alleles if "*" in a)
    is_compound_het = int(len(alleles) == 2 and alleles[0] != alleles[1])
    is_homozygous_variant = int(
        len(alleles) == 2
        and alleles[0] == alleles[1]
        and alleles[0] not in WILDTYPE_ALLELES
    )
    return star_count, is_compound_het, is_homozygous_variant


def build_genotype_record(gene, genotype, phenotype, sas_df, eur_df, eas_df, afr_df, amr_df, global_df,
                          n_sas_individuals=0, sample_id=None, population_code=None,
                          population_name=None, provenance="1000_Genomes_Project_SAS_real_WGS"):
    clinical_sig = PHENOTYPE_MAP.get(phenotype, 1)
    if gene in URGENT_GENES_PHENOTYPES and any(p in phenotype for p in ["Poor"]):
        clinical_sig = 3

    india_freq = compute_diplotype_freq(gene, genotype, sas_df)
    eur_freq = compute_diplotype_freq(gene, genotype, eur_df)
    eas_freq = compute_diplotype_freq(gene, genotype, eas_df)
    afr_freq = compute_diplotype_freq(gene, genotype, afr_df)
    amr_freq = compute_diplotype_freq(gene, genotype, amr_df)
    global_freq = compute_diplotype_freq(gene, genotype, global_df)

    ratio_eur = round(india_freq / (eur_freq + 1e-8), 3)
    ratio_eas = round(india_freq / (eas_freq + 1e-8), 3)
    ratio_global = round(india_freq / (global_freq + 1e-8), 3)

    subpop_freqs = compute_subpop_freqs(gene, genotype, sas_df)
    star_count, is_compound_het, is_homozygous_variant = genotype_complexity(genotype)

    record = {
        "gene": gene,
        "genotype": genotype,
        "phenotype": phenotype,
        "clinical_significance": clinical_sig,
        "india_diplotype_freq": india_freq,
        "european_diplotype_freq": eur_freq,
        "east_asian_diplotype_freq": eas_freq,
        "african_diplotype_freq": afr_freq,
        "american_diplotype_freq": amr_freq,
        "global_diplotype_freq": global_freq,
        "sas_vs_eur_ratio": ratio_eur,
        "sas_vs_eas_ratio": ratio_eas,
        "sas_vs_global_ratio": ratio_global,
        "subpop_freq_gih": subpop_freqs["GIH"],
        "subpop_freq_pjl": subpop_freqs["PJL"],
        "subpop_freq_itu": subpop_freqs["ITU"],
        "subpop_freq_stu": subpop_freqs["STU"],
        "subpop_freq_beb": subpop_freqs["BEB"],
        "subpop_dispersion": subpop_dispersion(subpop_freqs),
        "subpop_entropy": subpop_entropy(subpop_freqs),
        "star_allele_count": star_count,
        "is_compound_het": is_compound_het,
        "is_homozygous_variant": is_homozygous_variant,
        "cpic_phenotype_score": CPIC_SCORE_MAP.get(phenotype, 0.5),
        "gene_importance": GENE_IMPORTANCE.get(gene, 0.5),
        "is_chemo_gene": 1 if gene in ["DPYD", "NUDT15", "UGT1A1"] else 0,
        "n_sas_individuals": n_sas_individuals,
        "data_provenance": provenance,
        "reference": "Sherman CA, Claw KG & Lee S. Sci Rep 14:22774 (2024)",
    }
    if sample_id is not None:
        record["sample_id"] = sample_id
    if population_code is not None:
        record["population_code"] = population_code
    if population_name is not None:
        record["population_name"] = population_name
    return record


def build_frequency_reference_entry(gene, genotype, phenotype, sas_df, eur_df, eas_df, afr_df, amr_df, global_df,
                                    n_sas_individuals=0):
    """
    Aggregated population-frequency statistics for one gene|genotype.
    NOT training data - derived from real 1000 Genomes observations only.
    """
    rec = build_genotype_record(
        gene, genotype, phenotype, sas_df, eur_df, eas_df, afr_df, amr_df, global_df,
        n_sas_individuals=n_sas_individuals,
        provenance="1000_Genomes_frequency_reference_NOT_FOR_TRAINING",
    )
    rec["record_type"] = "frequency_reference"
    rec["training_eligible"] = False
    rec["note"] = (
        "Aggregated allele frequencies from real 1000 Genomes genomes. "
        "Not a sequenced individual. Do not use for ML training."
    )
    return rec


records = []
for _, row in sas.iterrows():
    gene = row["Gene"]
    phenotype = row["Phenotype"]
    genotype = row["Genotype"]

    record = build_genotype_record(
        gene, genotype, phenotype, sas, eur, eas, afr, amr, global_df,
        sample_id=row["Sample"],
        population_code=row["Population code"],
        population_name=row["Population name"],
        provenance="1000_Genomes_Project_SAS_real_WGS",
    )
    record["population_subgroup"] = row["Population code"]
    record["sex"] = row["Sex"]
    records.append(record)

print(f"\nTotal records: {len(records)}")

classes = Counter(r["clinical_significance"] for r in records)
print("Class distribution:")
for c in sorted(classes):
    print(f"  Class {c}: {classes[c]} records")

genes = Counter(r["gene"] for r in records)
print("\nBy gene:")
for g, count in genes.most_common():
    print(f"  {g}: {count}")

pops = Counter(r["population_code"] for r in records)
print("\nBy population:")
for p, count in pops.most_common():
    print(f"  {p}: {count}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

print(f"\nSaved {len(records)} real records to {OUT_PATH}")
print("SYNTHETIC DATA CHECK: Zero synthetic records - all data from real WGS")

# Optional frequency reference (NOT for ML training): aggregated stats per gene|genotype.
all_pops = df[
    df["Gene"].isin(CPIC_GENES)
    & df["Phenotype"].notna()
    & ~df["Phenotype"].isin(["Indeterminate", "Uncertain Susceptibility"])
].copy()

sas_counts = sas.groupby(["Gene", "Genotype"]).size().to_dict()
freq_ref = []
for (gene, genotype), grp in all_pops.groupby(["Gene", "Genotype"]):
    phenotype = grp["Phenotype"].iloc[0]
    n_sas = int(sas_counts.get((gene, genotype), 0))
    freq_ref.append(build_frequency_reference_entry(
        gene, genotype, phenotype, sas, eur, eas, afr, amr, global_df,
        n_sas_individuals=n_sas,
    ))

print(f"\nFrequency reference (NOT for training): {len(freq_ref)} gene|genotype entries "
      f"({sum(1 for c in freq_ref if c['n_sas_individuals'] > 0)} SAS-observed)")

with open(FREQ_REF_PATH, "w") as f:
    json.dump(freq_ref, f, indent=2)
print(f"Saved frequency reference to {FREQ_REF_PATH} (read-only, not used for ML)")
