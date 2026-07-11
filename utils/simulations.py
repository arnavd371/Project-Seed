from __future__ import annotations

import numpy as np

DEFAULT_N_SIM = 500
RNG = np.random.default_rng(42)


def bootstrap_frequency_shift(
    sas_freq: float,
    eur_freq: float,
    n_individuals: int = 489,
    n_sim: int = DEFAULT_N_SIM,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or RNG
    sas_counts = []
    eur_counts = []
    for _ in range(n_sim):
        sas_counts.append(int(rng.binomial(n_individuals, min(max(sas_freq, 0), 1))))
        eur_counts.append(int(rng.binomial(n_individuals, min(max(eur_freq, 0), 1))))
    sas_arr = np.array(sas_counts) / n_individuals
    eur_arr = np.array(eur_counts) / n_individuals
    return {
        "sas_freqs": sas_arr,
        "eur_freqs": eur_arr,
        "sas_mean": float(sas_arr.mean()),
        "eur_mean": float(eur_arr.mean()),
        "sas_ci": (float(np.percentile(sas_arr, 2.5)), float(np.percentile(sas_arr, 97.5))),
        "eur_ci": (float(np.percentile(eur_arr, 2.5)), float(np.percentile(eur_arr, 97.5))),
        "ratio_mean": float(np.mean(sas_arr / (eur_arr + 1e-8))),
        "n_sim": n_sim,
        "n_individuals": n_individuals,
    }


def gene_level_simulation(
    gene: str,
    genotypes: list[dict],
    n_sim: int = DEFAULT_N_SIM,
) -> dict:
    results = {}
    for g in genotypes:
        key = g["genotype"]
        sim = bootstrap_frequency_shift(
            g["sas_freq"],
            g["eur_freq"],
            n_individuals=g.get("n_individuals", 489),
            n_sim=n_sim,
        )
        results[key] = sim
    return {"gene": gene, "genotypes": results, "n_sim": n_sim}


def cyp2c19_clopidogrel_demo() -> dict:
    return {
        "title": "CYP2C19 *1/*2 - Clopidogrel (Intermediate Metabolizer)",
        "gene": "CYP2C19",
        "genotype": "*1/*2",
        "drug": "Clopidogrel",
        "clinical_note": (
            "Intermediate metabolizers have reduced CYP2C19 activity; clopidogrel "
            "pro-drug activation is impaired. More common in South Asian populations "
            "due to higher *2 allele frequency."
        ),
        "sas_freq": 0.28,
        "eur_freq": 0.12,
        "phenotype": "Intermediate Metabolizer",
        "urgency_class": 1,
        "simulation": gene_level_simulation(
            "CYP2C19",
            [
                {"genotype": "*1/*1", "sas_freq": 0.35, "eur_freq": 0.55},
                {"genotype": "*1/*2", "sas_freq": 0.28, "eur_freq": 0.12},
                {"genotype": "*2/*2", "sas_freq": 0.08, "eur_freq": 0.02},
            ],
        ),
    }


def dpyd_poor_metabolizer_demo() -> dict:
    genotype = "c.1905+1G>A (*2A)/c.1905+1G>A (*2A)"
    return {
        "title": "DPYD Poor Metabolizer - Fluoropyrimidine Toxicity (URGENT)",
        "gene": "DPYD",
        "genotype": genotype,
        "drug": "5-Fluorouracil / Capecitabine",
        "clinical_note": (
            "DPYD poor metabolizers cannot efficiently catabolise fluoropyrimidines, "
            "leading to life-threatening toxicity at standard doses. CPIC recommends "
            "urgent dose reduction or alternative therapy."
        ),
        "sas_freq": 0.001,
        "eur_freq": 0.012,
        "phenotype": "Poor Metabolizer",
        "urgency_class": 3,
        "simulation": gene_level_simulation(
            "DPYD",
            [
                {"genotype": "Reference/Reference", "sas_freq": 0.85, "eur_freq": 0.80},
                {"genotype": "Reference/c.85T>C (*9A)", "sas_freq": 0.10, "eur_freq": 0.12},
                {"genotype": genotype, "sas_freq": 0.001, "eur_freq": 0.012},
            ],
        ),
    }
