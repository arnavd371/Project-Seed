import json
import os

import numpy as np

import train_baseline
import train_model
import train_population_only
import train_risk_regressor
import validate_loso
import validate_gene_holdout
from phenotype_classifier import (
    build_baseline_features_from_entry,
    build_calibration_classifier_features_from_entry,
    build_group_keys,
    dedupe_records_by_group,
    grouped_cv_evaluate,
    make_classifier,
    metrics_at_group_level,
    population_adjusted_significance,
    rule_based_class,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def evaluate_rule_baseline(records):
    y = np.array([r["clinical_significance"] for r in records], dtype=int)
    groups = np.array(build_group_keys(records))
    y_pred = np.array([rule_based_class(r) for r in records], dtype=int)
    m = metrics_at_group_level(y, y_pred, groups)
    return {
        "accuracy": round(m["accuracy"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "per_class_f1": m["per_class_f1"],
        "n_groups": m["n_groups"],
        "cv_method": "rule-based (group-level, no ML training)",
        "description": "CPIC phenotype score thresholds only - traditional software baseline",
        "overfitting_diagnosis": {
            "train_accuracy": round(m["accuracy"], 4),
            "cv_accuracy": round(m["accuracy"], 4),
            "accuracy_gap": 0.0,
            "overfitting_flagged": False,
            "note": "Deterministic rule - no train/CV split applicable",
        },
    }


def build_overfitting_summary(baseline, calibration, population_only, risk):
    models = {
        "global_baseline": baseline.get("overfitting_diagnosis", {}),
        "india_calibrated": calibration.get("overfitting_diagnosis", {}),
        "population_only": population_only.get("overfitting_diagnosis", {}),
    }
    any_flagged = any(d.get("overfitting_flagged") for d in models.values())
    return {
        "models": models,
        "any_overfitting_flagged": any_flagged,
        "threshold": 0.10,
        "risk_regressor_note": (
            "Regression overfitting assessed via grouped-CV MAE during tuning "
            f"(MAE={risk.get('ml_mae', 'N/A')})"
        ),
    }


def build_table(baseline, rule, population_only, calibration, risk, loso, overfitting, gene_holdout):
    lines = []
    lines.append("SEED MODEL COMPARISON (SAS-only real WGS training)")
    lines.append("=" * 78)
    lines.append(f"{'Model':<32}{'Accuracy':<12}{'Macro F1':<12}{'Groups':<8}{'Label'}")
    lines.append("-" * 78)
    lines.append(f"{'Global Baseline (ML)':<32}{baseline['accuracy']:<12.2f}"
                 f"{baseline['macro_f1']:<12.2f}{baseline.get('n_groups', 168):<8}"
                 f"clinical_significance")
    lines.append(f"{'Rule Baseline (no ML)':<32}{rule['accuracy']:<12.2f}"
                 f"{rule['macro_f1']:<12.2f}{rule.get('n_groups', 168):<8}"
                 f"clinical_significance")
    lines.append(f"{'Population-Only ML (no CPIC)':<32}{population_only['accuracy']:<12.2f}"
                 f"{population_only['macro_f1']:<12.2f}{population_only.get('n_groups', 168):<8}"
                 f"population_adjusted (hard)")
    lines.append(f"{'Calibration ML (CPIC+pop)':<32}{calibration['accuracy']:<12.2f}"
                 f"{calibration['macro_f1']:<12.2f}{calibration.get('n_groups', 168):<8}"
                 f"population_adjusted")
    lines.append("=" * 78)
    lines.append("")
    lines.append("OVERFITTING DIAGNOSIS (train acc − grouped-CV acc, threshold 0.10)")
    for name, key in [
        ("Global Baseline", "global_baseline"),
        ("Population-Only ML", "population_only"),
        ("India Calibrated", "india_calibrated"),
    ]:
        d = overfitting["models"].get(key, {})
        flag = "FLAGGED" if d.get("overfitting_flagged") else "OK"
        lines.append(
            f"  {name}: train={d.get('train_accuracy', '?'):.3f}  "
            f"cv={d.get('cv_accuracy', '?'):.3f}  "
            f"gap={d.get('accuracy_gap', '?'):.3f}  [{flag}]"
        )
    lines.append("")
    lines.append("RISK REGRESSOR (continuous 0-1 India risk score)")
    lines.append(f"  Rule-adjusted MAE:    {risk.get('rule_adjusted_mae', risk.get('heuristic_mae', 0)):.4f}")
    lines.append(f"  Simple heuristic MAE: {risk['heuristic_mae']:.4f}")
    lines.append(f"  XGBRegressor MAE:     {risk['ml_mae']:.4f}  R²={risk['ml_r2']:.4f}")
    lines.append(f"  ML vs rule-adjusted:  {risk.get('mae_improvement_vs_rule', risk['mae_improvement']):+.4f} MAE")
    lines.append("")
    lines.append("LOSO (leave-one-SAS-subpopulation-out, group-level metrics)")
    for pop, r in loso.items():
        if pop == "summary":
            continue
        n_groups = r.get("n_test_groups", r.get("n_test", "?"))
        lines.append(f"  {pop} ({r['population_name']}): acc={r['accuracy']:.2f} "
                     f"macro_f1={r['macro_f1']:.2f} groups={n_groups}")
    lines.append(f"  Mean LOSO accuracy: {loso['summary']['mean_accuracy']:.2f} "
                 f"(±{loso['summary']['std_accuracy']:.2f})")
    lines.append("")
    lines.append("GENE-HELD-OUT CV (leave-one-gene-out, group-level metrics)")
    cal_gh = gene_holdout.get("calibration_model", {}).get("summary", {})
    pop_gh = gene_holdout.get("population_only_model", {}).get("summary", {})
    lines.append(
        f"  Calibration ML: mean acc={cal_gh.get('mean_accuracy', 0):.2f} "
        f"(±{cal_gh.get('std_accuracy', 0):.2f})  genes={cal_gh.get('n_genes', '?')}"
    )
    lines.append(
        f"  Population-Only ML: mean acc={pop_gh.get('mean_accuracy', 0):.2f} "
        f"(±{pop_gh.get('std_accuracy', 0):.2f})  genes={pop_gh.get('n_genes', '?')}"
    )
    lines.append("")
    lines.append(
        "Training: dedupe_records_by_group(real_training_data.json) ONLY - "
        "168 SAS genotype groups from 6,035 real individuals.\n"
        "No genotype_catalog.json. Grouped CV by gene|genotype. No SMOTE. "
        "Class weights (balanced) for India model."
    )
    return "\n".join(lines)


def main():
    with open(DATA_PATH) as f:
        all_records = json.load(f)
    records = dedupe_records_by_group(all_records)

    print("=" * 72)
    print(f"Evaluating on {len(records)} unique gene|genotype groups "
          f"(deduplicated from {len(all_records)} real SAS rows)\n")
    print("1/7 Global baseline...")
    baseline = train_baseline.main()
    print("\n" + "=" * 72)
    print("2/7 Population-only classifier (no CPIC)...")
    population_only = train_population_only.main()
    print("\n" + "=" * 72)
    print("3/7 Calibration classifier (CPIC + population)...")
    calibration = train_model.main()
    print("\n" + "=" * 72)
    print("4/7 Rule baseline (CPIC thresholds)...")
    rule = evaluate_rule_baseline(records)
    print(f"Rule baseline: acc={rule['accuracy']:.3f} macro_f1={rule['macro_f1']:.3f}")
    print("\n" + "=" * 72)
    print("5/7 Risk regressor...")
    risk = train_risk_regressor.main()
    print("\n" + "=" * 72)
    print("6/7 LOSO validation...")
    loso = validate_loso.main()
    print("\n" + "=" * 72)
    print("7/7 Gene-held-out CV...")
    gene_holdout = validate_gene_holdout.main()
    print("\n" + "=" * 72)

    overfitting = build_overfitting_summary(baseline, calibration, population_only, risk)
    table_text = build_table(
        baseline, rule, population_only, calibration, risk, loso, overfitting, gene_holdout
    )
    print(table_text)

    results = {
        "baseline": baseline,
        "rule_baseline": rule,
        "population_only": population_only,
        "india_calibrated": calibration,
        "risk_regressor": risk,
        "loso": loso,
        "gene_holdout": gene_holdout,
        "overfitting_diagnosis": overfitting,
        "accuracy_delta_india_vs_baseline": round(calibration["accuracy"] - baseline["accuracy"], 4),
        "methodological_notes": [
            "Real WGS only: dedupe_records_by_group(real_training_data.json) - "
            "168 unique gene|genotype groups from 6,035 SAS individuals. "
            "genotype_catalog.json is NOT used for training. "
            "Optional frequency_reference.json is read-only aggregated stats, not training data.",
            "Grouped CV: StratifiedGroupKFold by gene|genotype. Metrics at group level "
            "- each genotype counts once, not once per person.",
            "No SMOTE or synthetic oversampling. India classifier uses balanced "
            "class weights. Hyperparameters tuned optimizing grouped-CV macro F1.",
            "Overfitting check: train-set group-level accuracy vs grouped-CV accuracy; "
            "gap > 0.10 triggers stronger regularization (lower max_depth, higher "
            "min_child_weight/reg_lambda).",
            "India classifier uses population features + CPIC clinical_significance "
            "and predicts population_adjusted_significance.",
            "LOSO uses genotype-level SAS-wide frequencies identical across subpopulations "
            "for the same genotype - high LOSO accuracy reflects label consistency.",
        ],
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "comparison_results.json"), "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "comparison_table.txt"), "w") as f:
        f.write(table_text + "\n")

    print(f"\nSaved comparison_results.json and comparison_table.txt")
    return results


if __name__ == "__main__":
    main()
