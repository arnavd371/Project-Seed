"""
Shared feature schemas and XGBoost classifier factory for Seed's baseline
(European features) and India-calibrated models. Both use the same real SAS
1000 Genomes records so training and inference feature vectors stay aligned.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:  # scikit-learn < 1.1
    from sklearn.model_selection import GroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = False

NUM_CLASSES = 4  # clinical_significance in {0: No action, 1: Moderate, 2: Significant, 3: Urgent}

BASELINE_FEATURE_NAMES = [
    "european_diplotype_freq",
    "cpic_phenotype_score",
    "gene_importance",
    "is_chemo_gene",
]

INDIA_FEATURE_NAMES = [
    "india_diplotype_freq",
    "european_diplotype_freq",
    "sas_vs_eur_ratio",
    "cpic_phenotype_score",
    "gene_importance",
    "is_chemo_gene",
    "log_india_freq",
    "india_gt_europe",
    "freq_difference",
]

# Population-only features - no CPIC score (avoids label/feature circularity).
POPULATION_ONLY_FEATURE_NAMES = [
    "india_diplotype_freq",
    "european_diplotype_freq",
    "sas_vs_eur_ratio",
    "gene_importance",
    "is_chemo_gene",
    "log_india_freq",
    "india_gt_europe",
    "freq_difference",
]

EXPANDED_POPULATION_FEATURE_NAMES = [
    "india_diplotype_freq",
    "european_diplotype_freq",
    "east_asian_diplotype_freq",
    "african_diplotype_freq",
    "american_diplotype_freq",
    "global_diplotype_freq",
    "sas_vs_eur_ratio",
    "sas_vs_eas_ratio",
    "sas_vs_global_ratio",
    "gene_importance",
    "is_chemo_gene",
    "log_india_freq",
    "india_gt_europe",
    "freq_difference",
    "subpop_dispersion",
    "subpop_entropy",
    "star_allele_count",
    "is_compound_het",
    "is_homozygous_variant",
]

RISK_REGRESSOR_FEATURE_NAMES = EXPANDED_POPULATION_FEATURE_NAMES + ["clinical_significance"]

# At inference, clinical_significance comes from genotype→phenotype lookup (not leaked).
CALIBRATION_CLASSIFIER_FEATURE_NAMES = EXPANDED_POPULATION_FEATURE_NAMES + ["clinical_significance"]

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
}

DEFAULT_REGRESSOR_PARAMS = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
}

CLASSIFIER_PARAM_GRID = {
    "n_estimators": [80, 120, 160, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.08, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "min_child_weight": [1, 2, 3, 5],
    "gamma": [0.0, 0.1, 0.2, 0.5],
}

REGRESSOR_PARAM_GRID = {
    "n_estimators": [80, 120, 160, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.08, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "min_child_weight": [1, 2, 3],
    "gamma": [0.0, 0.1, 0.2],
}

# Stronger regularization grid - used when train/CV gap exceeds threshold.
REGULARIZED_CLASSIFIER_PARAM_GRID = {
    "n_estimators": [60, 80, 100, 120],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "reg_lambda": [2.0, 5.0, 10.0],
    "min_child_weight": [3, 5, 7],
    "gamma": [0.1, 0.2, 0.5],
}

REGULARIZED_REGRESSOR_PARAM_GRID = {
    "n_estimators": [60, 80, 100],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.08],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "reg_lambda": [2.0, 5.0, 10.0],
    "min_child_weight": [3, 5],
    "gamma": [0.1, 0.2, 0.5],
}

OVERFITTING_GAP_THRESHOLD = 0.10


def build_calibration_classifier_features_from_entry(entry):
    return build_expanded_features_from_entry(entry) + [float(entry["clinical_significance"])]


def build_risk_features_from_entry(entry):
    return build_expanded_features_from_entry(entry) + [float(entry["clinical_significance"])]


def rule_adjusted_risk(entry):
    adj = population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )
    return float(adj / 3.0)


def build_baseline_features_from_entry(entry):
    return [
        entry["european_diplotype_freq"],
        entry["cpic_phenotype_score"],
        entry["gene_importance"],
        entry["is_chemo_gene"],
    ]


def build_india_features_from_entry(entry):
    india_freq = entry["india_diplotype_freq"]
    eur_freq = entry["european_diplotype_freq"]
    return [
        india_freq,
        eur_freq,
        entry["sas_vs_eur_ratio"],
        entry["cpic_phenotype_score"],
        entry["gene_importance"],
        entry["is_chemo_gene"],
        float(np.log1p(india_freq)),
        1 if india_freq > eur_freq else 0,
        india_freq - eur_freq,
    ]


def build_population_features_from_entry(entry):
    india_freq = entry["india_diplotype_freq"]
    eur_freq = entry["european_diplotype_freq"]
    return [
        india_freq,
        eur_freq,
        entry["sas_vs_eur_ratio"],
        entry["gene_importance"],
        entry["is_chemo_gene"],
        float(np.log1p(india_freq)),
        1 if india_freq > eur_freq else 0,
        india_freq - eur_freq,
    ]


def build_expanded_features_from_entry(entry):
    india_freq = entry["india_diplotype_freq"]
    eur_freq = entry["european_diplotype_freq"]
    return [
        india_freq,
        eur_freq,
        entry.get("east_asian_diplotype_freq", 0.0),
        entry.get("african_diplotype_freq", 0.0),
        entry.get("american_diplotype_freq", 0.0),
        entry.get("global_diplotype_freq", 0.0),
        entry["sas_vs_eur_ratio"],
        entry.get("sas_vs_eas_ratio", 0.0),
        entry.get("sas_vs_global_ratio", 0.0),
        entry["gene_importance"],
        entry["is_chemo_gene"],
        float(np.log1p(india_freq)),
        1 if india_freq > eur_freq else 0,
        india_freq - eur_freq,
        entry.get("subpop_dispersion", 0.0),
        entry.get("subpop_entropy", 0.0),
        float(entry.get("star_allele_count", 0)),
        float(entry.get("is_compound_het", 0)),
        float(entry.get("is_homozygous_variant", 0)),
    ]


def population_adjusted_significance(clinical_sig, sas_ratio):
    """
    Ordinal label that requires population context - not derivable from CPIC alone.
    SAS-enriched clinically relevant genotypes bump urgency; SAS-depleted high-sig genotypes drop.
    """
    sig = int(clinical_sig)
    ratio = float(sas_ratio)
    if ratio >= 2.0 and sig >= 1:
        return min(3, sig + 1)
    if ratio >= 1.5 and sig >= 2:
        return min(3, sig + 1)
    if ratio <= 0.5 and sig >= 2:
        return max(0, sig - 1)
    return sig


def heuristic_india_risk(entry):
    """
    Simple fixed-formula risk - uses CPIC-derived clinical significance only,
    with a linear SAS-ratio bump. Does NOT apply full population-adjusted rules.
    Non-ML baseline for comparison against the risk regressor.
    """
    base = entry["clinical_significance"] / 3.0
    ratio = max(float(entry["sas_vs_eur_ratio"]), 0.0)
    correction = 1.0 + 0.12 * min(np.log1p(ratio), 1.5)
    return float(min(1.0, base * correction))


def india_risk_target(entry):
    adj = population_adjusted_significance(
        entry["clinical_significance"], entry["sas_vs_eur_ratio"]
    )
    return float(adj / 3.0)


def rule_based_class(entry):
    """
    CPIC phenotype score thresholds + gene-specific urgent overrides.
    Traditional software baseline - no ML, no population frequencies.
    """
    gene = entry.get("gene", "")
    phenotype = entry.get("phenotype", "")
    urgent_phenotypes = {"Poor Metabolizer", "Likely Poor Metabolizer"}
    if gene in {"DPYD", "NUDT15", "TPMT"} and phenotype in urgent_phenotypes:
        return 3

    score = float(entry["cpic_phenotype_score"])
    if score <= 0.1:
        return 2  # Poor metabolizer / poor function
    if score <= 0.6:
        return 1  # Intermediate / decreased function
    if score <= 1.2:
        return 0  # Normal metabolizer / normal function
    return 1  # Rapid / ultrarapid metabolizer


def make_risk_regressor(random_state=42, params=None):
    from xgboost import XGBRegressor
    cfg = {**DEFAULT_REGRESSOR_PARAMS, **(params or {})}
    return XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        **cfg,
    )


def make_classifier(random_state=42, params=None):
    cfg = {**DEFAULT_CLASSIFIER_PARAMS, **(params or {})}
    return XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        random_state=random_state,
        eval_metric="mlogloss",
        **cfg,
    )


def adaptive_n_splits(y, requested=5):
    """Cap k to smallest class count; prefer build_group_keys() + make_cv_splitter() for training."""
    counts = np.bincount(np.asarray(y))
    min_class_count = int(counts[counts > 0].min())
    return max(2, min(requested, min_class_count))


def dedupe_records_by_group(records):
    """One row per gene|genotype - required because ~33 duplicate rows per genotype skew row-level metrics."""
    seen = {}
    for r in records:
        key = f"{r['gene']}|{r['genotype']}"
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def metrics_at_group_level(y_true, y_pred, groups):
    """One vote per gene|genotype group when duplicate feature vectors exist per group."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    gy, gp = [], []
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        gy.append(int(y_true[mask][0]))
        gp.append(int(y_pred[mask][0]))

    gy = np.asarray(gy)
    gp = np.asarray(gp)
    labels = [0, 1, 2, 3]
    return {
        "n_groups": len(gy),
        "accuracy": float(accuracy_score(gy, gp)),
        "macro_f1": float(f1_score(gy, gp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gy, gp, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(gy, gp)),
        "per_class_f1": {
            str(c): float(f1_score(gy, gp, average=None, zero_division=0, labels=labels)[c])
            for c in labels
        },
        "per_class_precision": {
            str(c): float(precision_score(gy, gp, average=None, zero_division=0, labels=labels)[c])
            for c in labels
        },
        "per_class_recall": {
            str(c): float(recall_score(gy, gp, average=None, zero_division=0, labels=labels)[c])
            for c in labels
        },
    }


def grouped_cv_predict(X, y, groups, model_factory, requested_splits=5, use_class_weights=False):
    splitter, n_splits = make_cv_splitter(y, groups, requested=requested_splits)
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = model_factory()
        fit_kwargs = {}
        if use_class_weights:
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y[train_idx])
        model.fit(X[train_idx], y[train_idx], **fit_kwargs)
        y_pred[test_idx] = model.predict(X[test_idx])
    return y_pred, n_splits


def grouped_cv_evaluate(
    records, X, y, groups, model_factory, requested_splits=5, use_class_weights=False
):
    y_pred, n_splits = grouped_cv_predict(
        X, y, groups, model_factory, requested_splits, use_class_weights
    )

    group_m = metrics_at_group_level(y, y_pred, groups)
    group_m["n_splits"] = n_splits
    group_m["cv_method"] = "StratifiedGroupKFold by gene|genotype (metrics at group level)"
    group_m["row_level_accuracy"] = float(accuracy_score(y, y_pred))
    group_m["row_level_macro_f1"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
    if use_class_weights:
        group_m["class_weights"] = "balanced"
    return y_pred, group_m


def diagnose_overfitting(
    X, y, groups, model_factory, requested_splits=5, use_class_weights=False,
):
    _, cv_metrics = grouped_cv_evaluate(
        None, X, y, groups, model_factory, requested_splits, use_class_weights
    )

    model = model_factory()
    fit_kwargs = {}
    if use_class_weights:
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y)
    model.fit(X, y, **fit_kwargs)
    y_train_pred = model.predict(X)
    train_metrics = metrics_at_group_level(y, y_train_pred, groups)

    gap_acc = train_metrics["accuracy"] - cv_metrics["accuracy"]
    gap_f1 = train_metrics["macro_f1"] - cv_metrics["macro_f1"]

    return {
        "train_accuracy": round(train_metrics["accuracy"], 4),
        "train_macro_f1": round(train_metrics["macro_f1"], 4),
        "cv_accuracy": round(cv_metrics["accuracy"], 4),
        "cv_macro_f1": round(cv_metrics["macro_f1"], 4),
        "accuracy_gap": round(gap_acc, 4),
        "macro_f1_gap": round(gap_f1, 4),
        "overfitting_flagged": bool(gap_acc > OVERFITTING_GAP_THRESHOLD),
        "overfitting_threshold": OVERFITTING_GAP_THRESHOLD,
        "n_groups": train_metrics["n_groups"],
        "n_splits": cv_metrics["n_splits"],
    }


def tune_classifier_with_overfitting_guard(
    X, y, groups, model_factory_base, n_iter=30, use_class_weights=True,
):
    best_params, best_cv_f1, search_log = grouped_random_search(
        X, y, groups, CLASSIFIER_PARAM_GRID, model_factory_base,
        n_iter=n_iter, use_class_weights=use_class_weights,
    )

    def factory(params=best_params):
        return model_factory_base(params=params)

    diagnosis = diagnose_overfitting(X, y, groups, factory, use_class_weights=use_class_weights)
    regularization_applied = False

    if diagnosis["overfitting_flagged"]:
        reg_params, reg_cv_f1, reg_log = grouped_random_search(
            X, y, groups, REGULARIZED_CLASSIFIER_PARAM_GRID, model_factory_base,
            n_iter=n_iter, use_class_weights=use_class_weights,
        )
        reg_factory = lambda: model_factory_base(params=reg_params)
        reg_diagnosis = diagnose_overfitting(
            X, y, groups, reg_factory, use_class_weights=use_class_weights
        )
        if reg_diagnosis["accuracy_gap"] <= diagnosis["accuracy_gap"]:
            best_params = reg_params
            best_cv_f1 = reg_cv_f1
            search_log = reg_log
            diagnosis = reg_diagnosis
            regularization_applied = True

    return best_params, best_cv_f1, search_log, diagnosis, regularization_applied


def grouped_random_search(
    X, y, groups, param_grid, model_factory_base, n_iter=25, random_state=42,
    use_class_weights=True, requested_splits=5,
):
    rng = np.random.RandomState(random_state)
    best_score = -1.0
    best_params = None
    search_log = []

    for i in range(n_iter):
        trial = {k: rng.choice(v) for k, v in param_grid.items()}
        trial = {k: (v.item() if hasattr(v, "item") else v) for k, v in trial.items()}

        def factory(params=trial):
            return model_factory_base(params=params)

        _, metrics = grouped_cv_evaluate(
            None, X, y, groups, factory, requested_splits, use_class_weights
        )
        score = metrics["macro_f1"]
        search_log.append({"iter": i, "params": trial, "macro_f1": score})
        if score > best_score:
            best_score = score
            best_params = trial.copy()

    return best_params, best_score, search_log


def build_group_keys(records):
    """
    Group key per record: "GENE|GENOTYPE".

    Data leakage fix: features are (gene, genotype)-level population stats, but
    real_training_data.json has one row per individual (~33 duplicate rows per
    genotype). Plain row-level CV puts the same genotype in train and test,
    inflating accuracy to ~100%. Group-aware splitting keeps each genotype in
    one fold only.
    """
    return [f"{r['gene']}|{r['genotype']}" for r in records]


def group_stats(y, groups):
    """Per-class count of unique gene|genotype groups (not rows) - constrains meaningful fold count."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    stats = {}
    for c in sorted(set(y.tolist())):
        stats[int(c)] = len(set(groups[y == c].tolist()))
    return stats


def make_cv_splitter(y, groups, requested=5, random_state=42):
    """
    StratifiedGroupKFold (or GroupKFold fallback) so no genotype spans train and test.
    n_splits capped to unique group count. Class 3 ("Urgent") has only 2 genotype groups,
    so at most 2 folds will ever hold out a class-3 example - treat class-3 metrics accordingly.
    """
    n_unique_groups = len(set(np.asarray(groups).tolist()))
    n_splits = max(2, min(requested, n_unique_groups))

    if _HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = GroupKFold(n_splits=n_splits)

    return splitter, n_splits
