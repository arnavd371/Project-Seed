import json
import os

import numpy as np
from joblib import dump

from phenotype_classifier import (
    BASELINE_FEATURE_NAMES,
    CLASSIFIER_PARAM_GRID,
    build_baseline_features_from_entry,
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


def main():
    with open(DATA_PATH) as f:
        all_records = json.load(f)

    records = dedupe_records_by_group(all_records)
    X = np.array([build_baseline_features_from_entry(r) for r in records], dtype=float)
    y = np.array([r["clinical_significance"] for r in records], dtype=int)
    groups = np.array(build_group_keys(records))

    print(f"Group-level set: {len(records)} unique gene|genotype groups "
          f"(from {len(all_records)} real SAS rows)")
    print(f"Baseline features: {BASELINE_FEATURE_NAMES}")
    gstats = group_stats(y, groups)
    print(f"Groups per class: {gstats}")

    print("\nTuning hyperparameters (grouped random search, optimize CV macro F1)...")
    best_params, best_cv_f1, _, diagnosis, reg_applied = tune_classifier_with_overfitting_guard(
        X, y, groups,
        model_factory_base=lambda params: make_classifier(params=params),
        n_iter=25,
        use_class_weights=False,
    )
    print(f"Best grouped-CV macro F1: {best_cv_f1:.3f}")
    print(f"Tuned params: {best_params}")
    print(f"Overfitting diagnosis: train_acc={diagnosis['train_accuracy']:.3f} "
          f"cv_acc={diagnosis['cv_accuracy']:.3f} gap={diagnosis['accuracy_gap']:.3f} "
          f"flagged={diagnosis['overfitting_flagged']}")
    if reg_applied:
        print("Strong regularization applied after overfitting detected.")

    tuned_factory = lambda: make_classifier(params=best_params)
    _, metrics = grouped_cv_evaluate(records, X, y, groups, tuned_factory)

    print(f"\nGLOBAL BASELINE {metrics['n_splits']}-fold GROUPED CV (group-level):")
    print(f"  Accuracy: {metrics['accuracy']:.3f}  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"  Row-level accuracy (disclosure): {metrics['row_level_accuracy']:.3f}")

    model = make_classifier(params=best_params)
    model.fit(X, y)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    dump(model, os.path.join(ARTIFACTS_DIR, "baseline_xgb.pkl"))

    return {
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "per_class_f1": metrics["per_class_f1"],
        "n_splits": metrics["n_splits"],
        "n_groups": metrics["n_groups"],
        "row_level_accuracy": round(metrics["row_level_accuracy"], 4),
        "cv_method": metrics["cv_method"],
        "unique_groups_per_class": {str(k): v for k, v in gstats.items()},
        "tuned_params": best_params,
        "tuning_cv_macro_f1": round(best_cv_f1, 4),
        "overfitting_diagnosis": diagnosis,
        "regularization_applied": reg_applied,
    }


if __name__ == "__main__":
    main()
