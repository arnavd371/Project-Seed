import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from joblib import load

from phenotype_classifier import (
    BASELINE_FEATURE_NAMES,
    CALIBRATION_CLASSIFIER_FEATURE_NAMES,
    EXPANDED_POPULATION_FEATURE_NAMES,
    URGENCY_SHIFT_FEATURE_NAMES,
    URGENCY_SHIFT_LABELS,
    build_baseline_features_from_entry,
    build_calibration_classifier_features_from_entry,
    build_expanded_features_from_entry,
    build_risk_features_from_entry,
    build_urgency_shift_features_from_entry,
    population_adjusted_significance,
    heuristic_india_risk,
    rule_adjusted_risk,
    rule_based_class,
    dedupe_records_by_group,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
GENE_DRUG_MAP_PATH = os.path.join(SCRIPT_DIR, "..", "data", "gene_drug_map.json")
GNOMAD_PATH = os.path.join(SCRIPT_DIR, "..", "data", "gnomad_sas_reference.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")
COMBINED_CSV_PATH = os.path.expanduser("~/Downloads/1kgp-pgx-data/data/combined.csv")

CLASS_NAMES = {0: "No Action", 1: "Moderate", 2: "Significant", 3: "Urgent"}

SAS_POPULATIONS = ["GIH", "PJL", "ITU", "STU", "BEB"]
SAS_POPULATION_NAMES = {
    "GIH": "Gujarati Indians (Houston)",
    "PJL": "Punjabi (Lahore)",
    "ITU": "Indian Telugu (UK)",
    "STU": "Sri Lankan Tamil (UK)",
    "BEB": "Bengali (Bangladesh)",
}

EXCLUDED_GENES = {"VKORC1"}


def _model_fingerprint():
    names = [
        "india_calibrated_xgb.pkl",
        "baseline_xgb.pkl",
        "india_population_only_xgb.pkl",
        "india_risk_regressor.pkl",
        "india_urgency_shift_xgb.pkl",
        "feature_engineer.pkl",
    ]
    parts = []
    for name in names:
        path = os.path.join(ARTIFACTS_DIR, name)
        parts.append(os.path.getmtime(path) if os.path.exists(path) else 0)
    return tuple(parts)


def _feature_matrix(features, model, model_name):
    row = np.asarray(features, dtype=float).reshape(1, -1)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and row.shape[1] != expected:
        raise ValueError(
            f"{model_name} feature mismatch: expected {expected} features, got {row.shape[1]}. "
            "Restart the app or use Streamlit menu: Clear cache."
        )
    return row


class SeedScorer:
    def __init__(self):
        with open(DATA_PATH) as f:
            self.records = json.load(f)
        with open(GENE_DRUG_MAP_PATH) as f:
            self.gene_drug_map = json.load(f)["genes"]

        self.india_model = load(os.path.join(ARTIFACTS_DIR, "india_calibrated_xgb.pkl"))
        self.baseline_model = load(os.path.join(ARTIFACTS_DIR, "baseline_xgb.pkl"))
        pop_only_path = os.path.join(ARTIFACTS_DIR, "india_population_only_xgb.pkl")
        self.population_only_model = load(pop_only_path) if os.path.exists(pop_only_path) else None
        risk_path = os.path.join(ARTIFACTS_DIR, "india_risk_regressor.pkl")
        self.risk_regressor = load(risk_path) if os.path.exists(risk_path) else None
        shift_path = os.path.join(ARTIFACTS_DIR, "india_urgency_shift_xgb.pkl")
        self.urgency_shift_model = load(shift_path) if os.path.exists(shift_path) else None

        self.gnomad_ref = {}
        if os.path.exists(GNOMAD_PATH):
            with open(GNOMAD_PATH) as f:
                self.gnomad_ref = json.load(f).get("allele_frequencies", {})

        self.by_gene_genotype = defaultdict(list)
        self.gene_pop_totals = Counter()
        for r in self.records:
            self.by_gene_genotype[(r["gene"], r["genotype"])].append(r)
            self.gene_pop_totals[(r["gene"], r["population_code"])] += 1

        self.genes_available = sorted({
            g for (g, _) in self.by_gene_genotype if g not in EXCLUDED_GENES
        })

        self._rep_by_genotype = {
            (r["gene"], r["genotype"]): r
            for r in dedupe_records_by_group(self.records)
        }

        self._eas_df = None

    def _east_asian_df(self):
        if self._eas_df is None:
            if os.path.exists(COMBINED_CSV_PATH):
                full = pd.read_csv(COMBINED_CSV_PATH)
                self._eas_df = full[full["Superpopulation code"] == "EAS"]
            else:
                self._eas_df = pd.DataFrame(columns=["Gene", "Genotype"])
        return self._eas_df

    def east_asian_freq(self, gene, genotype):
        df = self._east_asian_df()
        gene_df = df[df["Gene"] == gene]
        total = len(gene_df)
        if total == 0:
            return 0.0
        count = len(gene_df[gene_df["Genotype"] == genotype])
        return round(count / total, 4)

    def genotypes_for_gene(self, gene):
        items = [
            (genotype, len(rows))
            for (g, genotype), rows in self.by_gene_genotype.items()
            if g == gene
        ]
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))

    def population_breakdown(self, gene, genotype):
        rows = self.by_gene_genotype.get((gene, genotype), [])
        counts = Counter(r["population_code"] for r in rows)
        breakdown = {}
        for pop in SAS_POPULATIONS:
            total = self.gene_pop_totals.get((gene, pop), 0)
            breakdown[pop] = round(100 * counts.get(pop, 0) / total, 1) if total else None
        return breakdown

    def gnomad_lookup(self, gene, genotype):
        alleles = [a.strip() for a in genotype.replace("/", "|").split("|") if a.strip()]
        hits = []
        for allele in alleles:
            key = f"{gene}|{allele}"
            if key in self.gnomad_ref:
                entry = self.gnomad_ref[key]
                hits.append({
                    "allele": allele,
                    "gnomad_sas_af": entry.get("gnomad_sas_af"),
                    "gnomad_nfe_af": entry.get("gnomad_nfe_af"),
                    "description": entry.get("description", ""),
                })
        return hits

    def _explain(self, model, features, feature_names):
        try:
            sys_path = os.path.join(SCRIPT_DIR, "..")
            if sys_path not in __import__("sys").path:
                __import__("sys").path.insert(0, sys_path)
            from utils.shap_explainer import explain_prediction
            return explain_prediction(model, features, feature_names)
        except Exception:
            return {"method": "none", "contributions": [], "top_positive": [], "top_negative": []}

    def bootstrap_freq_ci(self, gene, genotype, n_bootstrap=500, alpha=0.05):
        rows = self.by_gene_genotype.get((gene, genotype), [])
        if len(rows) < 5:
            freq = rows[0]["india_diplotype_freq"] if rows else 0.0
            return {"point": freq, "ci_low": freq, "ci_high": freq, "n": len(rows)}

        rng = np.random.default_rng(42)
        freqs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(rows), size=len(rows))
            sample = [rows[i] for i in idx]
            freqs.append(len(sample) / max(len(rows), 1) * rows[0]["india_diplotype_freq"])

        freqs = np.array(freqs)
        lo = float(np.percentile(freqs, 100 * alpha / 2))
        hi = float(np.percentile(freqs, 100 * (1 - alpha / 2)))
        return {
            "point": rows[0]["india_diplotype_freq"],
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "n": len(rows),
        }

    def score(self, gene, genotype):
        rows = self.by_gene_genotype.get((gene, genotype))
        if not rows:
            raise KeyError(
                f"No real SAS individual for {gene} {genotype}. "
                f"Only genotypes observed in the 1000 Genomes SAS reference are supported."
            )
        rep = dict(self._rep_by_genotype[(gene, genotype)])
        rep["n_real_individuals"] = len(rows)

        pop_x = build_calibration_classifier_features_from_entry(rep)
        baseline_x = build_baseline_features_from_entry(rep)
        pop_only_x = build_expanded_features_from_entry(rep)

        cal_X = _feature_matrix(pop_x, self.india_model, "Calibration ML")
        base_X = _feature_matrix(baseline_x, self.baseline_model, "Global baseline")
        india_proba = self.india_model.predict_proba(cal_X)[0]
        baseline_proba = self.baseline_model.predict_proba(base_X)[0]

        if self.population_only_model is not None:
            pop_X = _feature_matrix(
                pop_only_x, self.population_only_model, "Population-only ML"
            )
            pop_only_proba = self.population_only_model.predict_proba(pop_X)[0]
            pop_only_pred = int(self.population_only_model.predict(pop_X)[0])
            pop_only_conf = round(float(pop_only_proba[pop_only_pred]), 3)
        else:
            pop_only_proba = None
            pop_only_pred = None
            pop_only_conf = None

        india_expected = float(sum(i * p for i, p in enumerate(india_proba)))
        baseline_expected = float(sum(i * p for i, p in enumerate(baseline_proba)))

        heuristic_risk = round(heuristic_india_risk(rep), 3)
        rule_risk = round(rule_adjusted_risk(rep), 3)
        if self.risk_regressor is not None:
            risk_x = build_risk_features_from_entry(rep)
            risk_X = _feature_matrix(risk_x, self.risk_regressor, "Risk regressor")
            ml_risk = round(float(self.risk_regressor.predict(risk_X)[0]), 3)
            india_risk_score = rule_risk
        else:
            ml_risk = None
            india_risk_score = rule_risk

        baseline_risk_score = round(baseline_expected / 3.0, 3)
        rule_pred = rule_based_class(rep)
        adjusted_sig = population_adjusted_significance(
            rep["clinical_significance"], rep["sas_vs_eur_ratio"]
        )

        india_pred = int(self.india_model.predict(cal_X)[0])
        baseline_pred = int(self.baseline_model.predict(base_X)[0])
        freq_ci = self.bootstrap_freq_ci(gene, genotype)

        gene_info = self.gene_drug_map.get(gene, {})
        gnomad_hits = self.gnomad_lookup(gene, genotype)
        calibration_explain = self._explain(
            self.india_model, pop_x, CALIBRATION_CLASSIFIER_FEATURE_NAMES
        )
        pop_only_explain = (
            self._explain(self.population_only_model, pop_only_x, EXPANDED_POPULATION_FEATURE_NAMES)
            if self.population_only_model is not None else None
        )

        if self.urgency_shift_model is not None:
            shift_x = build_urgency_shift_features_from_entry(rep)
            shift_X = _feature_matrix(shift_x, self.urgency_shift_model, "India Urgency Shift")
            shift_proba = self.urgency_shift_model.predict_proba(shift_X)[0]
            shift_pred = int(self.urgency_shift_model.predict(shift_X)[0])
            # binary:logistic may return shape (1,2) or classes_[pred]
            if len(shift_proba) == 2:
                shift_conf = round(float(shift_proba[shift_pred]), 3)
                shift_prob_higher = round(float(shift_proba[1]), 3)
            else:
                shift_conf = round(float(shift_proba[0]), 3)
                shift_prob_higher = shift_conf if shift_pred == 1 else round(1.0 - shift_conf, 3)
            shift_label = URGENCY_SHIFT_LABELS.get(shift_pred, str(shift_pred))
            shift_explain = self._explain(
                self.urgency_shift_model, shift_x, URGENCY_SHIFT_FEATURE_NAMES
            )
        else:
            shift_pred = None
            shift_conf = None
            shift_prob_higher = None
            shift_label = None
            shift_explain = None

        return {
            "gene": gene,
            "genotype": genotype,
            "phenotype": rep["phenotype"],
            "observed_clinical_significance": rep["clinical_significance"],
            "observed_clinical_significance_label": CLASS_NAMES[rep["clinical_significance"]],
            "population_adjusted_significance": adjusted_sig,
            "population_adjusted_label": CLASS_NAMES[adjusted_sig],
            "india_diplotype_freq": rep["india_diplotype_freq"],
            "india_freq_ci_low": freq_ci["ci_low"],
            "india_freq_ci_high": freq_ci["ci_high"],
            "european_diplotype_freq": rep["european_diplotype_freq"],
            "east_asian_diplotype_freq": self.east_asian_freq(gene, genotype),
            "sas_vs_eur_ratio": rep["sas_vs_eur_ratio"],
            "cpic_phenotype_score": rep["cpic_phenotype_score"],
            "gene_importance": rep["gene_importance"],
            "is_chemo_gene": bool(rep["is_chemo_gene"]),
            "n_real_individuals": len(rows),
            "sas_observed": True,
            "india_risk_score": india_risk_score,
            "heuristic_risk_score": heuristic_risk,
            "rule_adjusted_risk_score": rule_risk,
            "ml_risk_score": ml_risk,
            "baseline_risk_score": baseline_risk_score,
            "risk_score_delta": round(india_risk_score - baseline_risk_score, 3),
            "india_predicted_class": india_pred,
            "india_predicted_label": CLASS_NAMES[india_pred],
            "india_confidence": round(float(india_proba[india_pred]), 3),
            "baseline_predicted_class": baseline_pred,
            "baseline_predicted_label": CLASS_NAMES[baseline_pred],
            "baseline_confidence": round(float(baseline_proba[baseline_pred]), 3),
            "population_only_predicted_class": pop_only_pred,
            "population_only_predicted_label": CLASS_NAMES[pop_only_pred] if pop_only_pred is not None else None,
            "population_only_confidence": pop_only_conf,
            "population_only_available": self.population_only_model is not None,
            "dual_model_agreement": (
                pop_only_pred == india_pred if pop_only_pred is not None else None
            ),
            "urgency_shift_predicted_class": shift_pred,
            "urgency_shift_label": shift_label,
            "urgency_shift_confidence": shift_conf,
            "urgency_shift_prob_higher": shift_prob_higher,
            "urgency_shift_available": self.urgency_shift_model is not None,
            "urgency_shift_rule_higher": bool(adjusted_sig > rep["clinical_significance"]),
            "urgency_shift_explanations": shift_explain,
            "calibration_explanations": calibration_explain,
            "population_only_explanations": pop_only_explain,
            "gnomad_comparison": gnomad_hits,
            "rule_predicted_class": rule_pred,
            "rule_predicted_label": CLASS_NAMES[rule_pred],
            "classification_changed": bool(india_pred != baseline_pred),
            "population_breakdown": self.population_breakdown(gene, genotype),
            "full_name": gene_info.get("full_name", gene),
            "drugs": gene_info.get("drugs", []),
            "cpic_level": gene_info.get("cpic_level", "N/A"),
            "india_relevance_note": gene_info.get("india_relevance_note"),
            "data_provenance": rep.get("data_provenance", "1000_Genomes_Project_SAS_real_WGS"),
            "reference": rep["reference"],
        }


def score_genotype(gene, genotype):
    return SeedScorer().score(gene, genotype)
