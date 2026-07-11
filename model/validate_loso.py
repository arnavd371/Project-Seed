import json
import os

import numpy as np

from phenotype_classifier import (
    build_calibration_classifier_features_from_entry,
    population_adjusted_significance,
    make_classifier,
    build_group_keys,
    metrics_at_group_level,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")

SAS_POPS = ["GIH", "PJL", "ITU", "STU", "BEB"]
POP_NAMES = {
    "GIH": "Gujarati (Houston)",
    "PJL": "Punjabi (Lahore)",
    "ITU": "Indian Telugu",
    "STU": "Sri Lankan Tamil",
    "BEB": "Bengali",
}


def adjusted_label(entry):
    return population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )


def main():
    with open(DATA_PATH) as f:
        records = json.load(f)

    results = {}
    for held_out in SAS_POPS:
        train = [r for r in records if r["population_code"] != held_out]
        test = [r for r in records if r["population_code"] == held_out]
        if not test:
            continue

        X_train = np.array([build_calibration_classifier_features_from_entry(r) for r in train], dtype=float)
        y_train = np.array([adjusted_label(r) for r in train], dtype=int)
        X_test = np.array([build_calibration_classifier_features_from_entry(r) for r in test], dtype=float)
        y_test = np.array([adjusted_label(r) for r in test], dtype=int)

        model = make_classifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_groups = np.array(build_group_keys(test))

        m = metrics_at_group_level(y_test, y_pred, test_groups)
        acc = m["accuracy"]
        macro_f1 = m["macro_f1"]
        results[held_out] = {
            "population_name": POP_NAMES[held_out],
            "n_test_rows": len(test),
            "n_test_groups": m["n_groups"],
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
        }
        print(f"LOSO held-out {held_out}: groups={m['n_groups']} acc={acc:.3f} macro_f1={macro_f1:.3f}")

    accs = [r["accuracy"] for r in results.values()]
    results["summary"] = {
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "n_folds": len(accs),
    }
    print(f"\nLOSO mean accuracy: {results['summary']['mean_accuracy']:.3f} "
          f"(±{results['summary']['std_accuracy']:.3f})")
    return results


if __name__ == "__main__":
    main()
