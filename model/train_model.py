import json
import os

import numpy as np
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight

from phenotype_classifier import (
    CALIBRATION_CLASSIFIER_FEATURE_NAMES,
    build_calibration_classifier_features_from_entry,
    population_adjusted_significance,
    build_group_keys,
    dedupe_records_by_group,
    group_stats,
    grouped_cv_evaluate,
    make_classifier,
    tune_classifier_with_overfitting_guard,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")

CLASS_NAMES = {0: "No action", 1: "Moderate", 2: "Significant", 3: "Urgent"}


def adjusted_label(entry):
    return population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )


def load_training_records():
    with open(DATA_PATH) as f:
        all_records = json.load(f)
    return dedupe_records_by_group(all_records), "sas_deduped"


def main():
    records, source = load_training_records()
    X = np.array([build_calibration_classifier_features_from_entry(r) for r in records], dtype=float)
    y = np.array([adjusted_label(r) for r in records], dtype=int)
    groups = np.array(build_group_keys(records))

    print(f"Training source: {source} (real SAS WGS only - no catalog)")
    print(f"Unique gene|genotype groups: {len(records)} from real_training_data.json")
    print(f"Features ({len(CALIBRATION_CLASSIFIER_FEATURE_NAMES)}): "
          f"19 population + clinical_significance")
    print(f"Label: population_adjusted_significance")

    gstats = group_stats(y, groups)
    print(f"Groups per class: {gstats}")

    print("\nTuning hyperparameters (grouped random search, optimize CV macro F1)...")
    best_params, best_cv_f1, _, diagnosis, reg_applied = tune_classifier_with_overfitting_guard(
        X, y, groups,
        model_factory_base=lambda params: make_classifier(params=params),
        n_iter=30,
        use_class_weights=True,
    )
    print(f"Best grouped-CV macro F1: {best_cv_f1:.3f}")
    print(f"Tuned params: {best_params}")
    print(f"Overfitting diagnosis: train_acc={diagnosis['train_accuracy']:.3f} "
          f"cv_acc={diagnosis['cv_accuracy']:.3f} gap={diagnosis['accuracy_gap']:.3f} "
          f"flagged={diagnosis['overfitting_flagged']}")
    if reg_applied:
        print("Strong regularization applied after overfitting detected.")

    tuned_factory = lambda: make_classifier(params=best_params)
    _, metrics = grouped_cv_evaluate(
        records, X, y, groups, tuned_factory, use_class_weights=True
    )

    print(f"\nSAS-ONLY {metrics['n_splits']}-fold GROUPED CV (group-level):")
    print(f"  Accuracy: {metrics['accuracy']:.3f}  macro F1: {metrics['macro_f1']:.3f}  "
          f"({metrics['n_groups']} groups)")
    for c in [0, 1, 2, 3]:
        print(f"  Class {c} ({CLASS_NAMES[c]}): F1={metrics['per_class_f1'][str(c)]:.3f}")

    model = make_classifier(params=best_params)
    sw = compute_sample_weight("balanced", y)
    model.fit(X, y, sample_weight=sw)
    importances = dict(zip(CALIBRATION_CLASSIFIER_FEATURE_NAMES, model.feature_importances_.tolist()))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    dump(model, os.path.join(ARTIFACTS_DIR, "india_calibrated_xgb.pkl"))
    dump(
        {
            "feature_names": CALIBRATION_CLASSIFIER_FEATURE_NAMES,
            "label": "population_adjusted_significance",
            "build_fn_name": "build_calibration_classifier_features_from_entry",
            "training_source": source,
            "n_groups": len(records),
            "tuned_params": best_params,
            "cv_method": metrics["cv_method"],
            "n_splits_used": metrics["n_splits"],
            "unique_groups_per_class": {str(k): v for k, v in gstats.items()},
            "overfitting_diagnosis": diagnosis,
            "regularization_applied": reg_applied,
        },
        os.path.join(ARTIFACTS_DIR, "feature_engineer.pkl"),
    )

    result = {
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "cohen_kappa": round(metrics["cohen_kappa"], 4),
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
        "n_splits": metrics["n_splits"],
        "n_groups": metrics["n_groups"],
        "training_source": source,
        "row_level_accuracy": round(metrics["row_level_accuracy"], 4),
        "cv_method": metrics["cv_method"],
        "unique_groups_per_class": {str(k): v for k, v in gstats.items()},
        "feature_importances": importances,
        "features": CALIBRATION_CLASSIFIER_FEATURE_NAMES,
        "label": "population_adjusted_significance",
        "tuned_params": best_params,
        "tuning_cv_macro_f1": round(best_cv_f1, 4),
        "class_weights": "balanced",
        "overfitting_diagnosis": diagnosis,
        "regularization_applied": reg_applied,
    }
    return result


if __name__ == "__main__":
    main()
