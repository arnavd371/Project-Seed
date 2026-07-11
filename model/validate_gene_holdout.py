import json
import os

import numpy as np

from phenotype_classifier import (
    build_calibration_classifier_features_from_entry,
    build_expanded_features_from_entry,
    population_adjusted_significance,
    make_classifier,
    build_group_keys,
    metrics_at_group_level,
    dedupe_records_by_group,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")


def adjusted_label(entry):
    return population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )


def gene_holdout_cv(records, feature_fn, label_fn):
    genes = sorted(set(r["gene"] for r in records))
    results = {}

    for held_gene in genes:
        train = [r for r in records if r["gene"] != held_gene]
        test = [r for r in records if r["gene"] == held_gene]
        if len(test) < 2:
            continue

        X_train = np.array([feature_fn(r) for r in train], dtype=float)
        y_train = np.array([label_fn(r) for r in train], dtype=int)
        X_test = np.array([feature_fn(r) for r in test], dtype=float)
        y_test = np.array([label_fn(r) for r in test], dtype=int)
        test_groups = np.array(build_group_keys(test))

        model = make_classifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = metrics_at_group_level(y_test, y_pred, test_groups)
        results[held_gene] = {
            "n_test_groups": m["n_groups"],
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
        }
        print(f"  Held out {held_gene}: groups={m['n_groups']} acc={m['accuracy']:.3f}")

    accs = [r["accuracy"] for r in results.values()]
    summary = {
        "mean_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        "std_accuracy": round(float(np.std(accs)), 4) if accs else 0.0,
        "n_genes": len(results),
    }
    return {"per_gene": results, "summary": summary}


def main():
    with open(DATA_PATH) as f:
        records = dedupe_records_by_group(json.load(f))

    print("Gene-held-out CV - Calibration model (CPIC + population features)")
    calibration = gene_holdout_cv(
        records, build_calibration_classifier_features_from_entry, adjusted_label
    )
    print(f"  Mean accuracy: {calibration['summary']['mean_accuracy']:.3f}")

    print("\nGene-held-out CV - Population-only model (no CPIC)")
    pop_only = gene_holdout_cv(
        records, build_expanded_features_from_entry, adjusted_label
    )
    print(f"  Mean accuracy: {pop_only['summary']['mean_accuracy']:.3f}")

    return {"calibration_model": calibration, "population_only_model": pop_only}


if __name__ == "__main__":
    main()
