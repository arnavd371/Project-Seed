import json
import os

import numpy as np
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight

from phenotype_classifier import (
    EXPANDED_POPULATION_FEATURE_NAMES,
    CLASSIFIER_PARAM_GRID,
    build_expanded_features_from_entry,
    population_adjusted_significance,
    build_group_keys,
    dedupe_records_by_group,
    group_stats,
    grouped_cv_evaluate,
    grouped_random_search,
    make_classifier,
    diagnose_overfitting,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")


def adjusted_label(entry):
    return population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )


def main():
    with open(DATA_PATH) as f:
        records = dedupe_records_by_group(json.load(f))

    X = np.array([build_expanded_features_from_entry(r) for r in records], dtype=float)
    y = np.array([adjusted_label(r) for r in records], dtype=int)
    groups = np.array(build_group_keys(records))

    print(f"Population-only model: {len(records)} SAS genotype groups, "
          f"{len(EXPANDED_POPULATION_FEATURE_NAMES)} features (no CPIC)")

    best_params, best_cv_f1, _ = grouped_random_search(
        X, y, groups, CLASSIFIER_PARAM_GRID,
        model_factory_base=lambda params: make_classifier(params=params),
        n_iter=25,
        use_class_weights=True,
    )

    tuned_factory = lambda: make_classifier(params=best_params)
    _, metrics = grouped_cv_evaluate(
        records, X, y, groups, tuned_factory, use_class_weights=True
    )

    model = make_classifier(params=best_params)
    sw = compute_sample_weight("balanced", y)
    model.fit(X, y, sample_weight=sw)
    overfit = diagnose_overfitting(X, y, groups, tuned_factory, use_class_weights=True)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    dump(model, os.path.join(ARTIFACTS_DIR, "india_population_only_xgb.pkl"))

    print(f"CV accuracy: {metrics['accuracy']:.3f}  macro F1: {metrics['macro_f1']:.3f}")

    return {
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "cohen_kappa": round(metrics["cohen_kappa"], 4),
        "per_class_f1": metrics["per_class_f1"],
        "n_groups": metrics["n_groups"],
        "n_splits": metrics.get("n_splits", 5),
        "cv_method": metrics["cv_method"],
        "features": EXPANDED_POPULATION_FEATURE_NAMES,
        "label": "population_adjusted_significance",
        "description": "No CPIC/clinical_significance in features - population structure only",
        "tuned_params": best_params,
        "overfitting_diagnosis": overfit,
    }


if __name__ == "__main__":
    main()
