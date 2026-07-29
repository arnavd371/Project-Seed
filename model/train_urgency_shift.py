import json
import os

import numpy as np
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight

from phenotype_classifier import (
    CLASSIFIER_PARAM_GRID,
    URGENCY_SHIFT_FEATURE_NAMES,
    URGENCY_SHIFT_LABELS,
    build_group_keys,
    build_urgency_shift_features_from_entry,
    dedupe_records_by_group,
    diagnose_overfitting,
    grouped_cv_evaluate,
    grouped_random_search,
    make_classifier,
    urgency_shift_label,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")


def main():
    with open(DATA_PATH) as f:
        records = dedupe_records_by_group(json.load(f))

    X = np.array([build_urgency_shift_features_from_entry(r) for r in records], dtype=float)
    y = np.array([urgency_shift_label(r) for r in records], dtype=int)
    groups = np.array(build_group_keys(records))

    print(
        f"India Urgency Shift model: {len(records)} genotype groups, "
        f"{len(URGENCY_SHIFT_FEATURE_NAMES)} features (no CPIC class)"
    )
    print(f"Label balance: higher={int(y.sum())}  no_shift={int((y == 0).sum())}")

    best_params, best_cv_f1, _ = grouped_random_search(
        X,
        y,
        groups,
        CLASSIFIER_PARAM_GRID,
        model_factory_base=lambda params: make_classifier(params=params, num_classes=2),
        n_iter=30,
        use_class_weights=True,
    )

    tuned_factory = lambda: make_classifier(params=best_params, num_classes=2)
    _, metrics = grouped_cv_evaluate(
        records, X, y, groups, tuned_factory, use_class_weights=True
    )

    model = make_classifier(params=best_params, num_classes=2)
    sw = compute_sample_weight("balanced", y)
    model.fit(X, y, sample_weight=sw)
    overfit = diagnose_overfitting(X, y, groups, tuned_factory, use_class_weights=True)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, "india_urgency_shift_xgb.pkl")
    dump(model, out_path)

    result = {
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics.get("weighted_f1", metrics["macro_f1"]), 4),
        "cohen_kappa": round(metrics.get("cohen_kappa", 0.0), 4),
        "per_class_f1": metrics.get("per_class_f1", {}),
        "n_groups": metrics["n_groups"],
        "n_splits": metrics.get("n_splits", 5),
        "cv_method": metrics["cv_method"],
        "features": URGENCY_SHIFT_FEATURE_NAMES,
        "labels": URGENCY_SHIFT_LABELS,
        "label": "india_urgency_higher_than_cpic",
        "description": (
            "Binary XGBoost: predict whether population-adjusted urgency is higher "
            "than CPIC clinical_significance alone, from population structure features."
        ),
        "tuned_params": best_params,
        "tuning_cv_macro_f1": round(float(best_cv_f1), 4),
        "overfitting_diagnosis": overfit,
        "artifact": "india_urgency_shift_xgb.pkl",
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "urgency_shift_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {out_path}")
    print(f"CV accuracy: {result['accuracy']:.3f}  macro F1: {result['macro_f1']:.3f}")
    print(f"Overfit gap: {overfit.get('accuracy_gap', 'n/a')}")
    return result


if __name__ == "__main__":
    main()
