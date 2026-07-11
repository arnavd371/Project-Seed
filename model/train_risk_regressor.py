import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from phenotype_classifier import (
    RISK_REGRESSOR_FEATURE_NAMES,
    REGRESSOR_PARAM_GRID,
    build_risk_features_from_entry,
    heuristic_india_risk,
    rule_adjusted_risk,
    india_risk_target,
    build_group_keys,
    dedupe_records_by_group,
    make_risk_regressor,
    make_cv_splitter,
    grouped_random_search,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")


def grouped_cv_regression(X, y, groups, regressor_factory, requested_splits=5):
    y_bins = np.digitize(y, bins=[0.25, 0.5, 0.75])
    splitter, n_splits = make_cv_splitter(y_bins, groups, requested=requested_splits)
    preds = np.zeros_like(y)
    for train_idx, test_idx in splitter.split(X, y_bins, groups):
        reg = regressor_factory()
        reg.fit(X[train_idx], y[train_idx])
        preds[test_idx] = reg.predict(X[test_idx])
    return preds, n_splits


def main():
    with open(DATA_PATH) as f:
        all_records = json.load(f)
    records = dedupe_records_by_group(all_records)

    X = np.array([build_risk_features_from_entry(r) for r in records], dtype=float)
    y = np.array([india_risk_target(r) for r in records], dtype=float)
    groups = np.array(build_group_keys(records))

    rule_preds = np.array([rule_adjusted_risk(r) for r in records], dtype=float)
    rule_mae = mean_absolute_error(y, rule_preds)
    heuristic_preds = np.array([heuristic_india_risk(r) for r in records], dtype=float)
    heuristic_mae = mean_absolute_error(y, heuristic_preds)
    heuristic_rmse = float(np.sqrt(mean_squared_error(y, heuristic_preds)))

    print("Tuning risk regressor (grouped random search on MAE)...")
    best_params = None
    best_mae = float("inf")
    rng = np.random.RandomState(42)
    for i in range(20):
        trial = {k: rng.choice(v) for k, v in REGRESSOR_PARAM_GRID.items()}
        preds, _ = grouped_cv_regression(
            X, y, groups, lambda p=trial: make_risk_regressor(params=p)
        )
        mae = mean_absolute_error(y, preds)
        if mae < best_mae:
            best_mae = mae
            best_params = trial.copy()
    print(f"Best grouped-CV MAE during search: {best_mae:.4f}")
    print(f"Tuned params: {best_params}")

    ml_preds, n_splits = grouped_cv_regression(
        X, y, groups, lambda: make_risk_regressor(params=best_params)
    )
    ml_mae = mean_absolute_error(y, ml_preds)
    ml_rmse = float(np.sqrt(mean_squared_error(y, ml_preds)))
    ml_r2 = r2_score(y, ml_preds)

    print(f"Risk regressor - grouped {n_splits}-fold CV (n={len(records)} groups)")
    print(f"  Rule-adjusted MAE:     {rule_mae:.4f}")
    print(f"  Simple heuristic MAE:  {heuristic_mae:.4f}")
    print(f"  XGBRegressor MAE:      {ml_mae:.4f}  R²: {ml_r2:.4f}")
    print(f"  ML vs rule-adjusted:   {(rule_mae - ml_mae):+.4f} MAE")

    reg = make_risk_regressor(params=best_params)
    reg.fit(X, y)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    dump(reg, os.path.join(ARTIFACTS_DIR, "india_risk_regressor.pkl"))

    importances = dict(zip(RISK_REGRESSOR_FEATURE_NAMES, reg.feature_importances_.tolist()))
    return {
        "rule_adjusted_mae": round(float(rule_mae), 4),
        "heuristic_mae": round(float(heuristic_mae), 4),
        "heuristic_rmse": round(float(heuristic_rmse), 4),
        "ml_mae": round(float(ml_mae), 4),
        "ml_rmse": round(float(ml_rmse), 4),
        "ml_r2": round(float(ml_r2), 4),
        "mae_improvement_vs_rule": round(float(rule_mae - ml_mae), 4),
        "mae_improvement": round(float(heuristic_mae - ml_mae), 4),
        "n_splits": n_splits,
        "feature_importances": importances,
        "tuned_params": best_params,
    }


if __name__ == "__main__":
    main()
