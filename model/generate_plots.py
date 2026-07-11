import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from joblib import load

from phenotype_classifier import (
    EXPANDED_POPULATION_FEATURE_NAMES,
    CALIBRATION_CLASSIFIER_FEATURE_NAMES,
    build_calibration_classifier_features_from_entry,
    population_adjusted_significance,
    build_group_keys,
    dedupe_records_by_group,
    make_classifier,
    grouped_cv_predict,
    metrics_at_group_level,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "real_training_data.json")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")
CSV_PATH = os.path.expanduser("~/Downloads/1kgp-pgx-data/data/combined.csv")

NAVY = "#0a0f1e"
PANEL = "#131a2e"
TEAL = "#00d4aa"
GREY = "#7a8699"
LIGHT_BLUE = "#5da9e9"
TEXT = "#e8ecf4"

CLASS_LABELS = ["No Action", "Moderate", "Significant", "Urgent"]


def _dark_axes(fig, ax):
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GREY)


def plot_learning_curve(records, out_path):
    y = np.array([
        population_adjusted_significance(r["clinical_significance"], r["sas_vs_eur_ratio"])
        for r in records
    ])
    groups = np.array(build_group_keys(records))
    X = np.array([build_calibration_classifier_features_from_entry(r) for r in records], dtype=float)

    unique_groups = np.array(sorted(set(groups.tolist())))
    rng = np.random.RandomState(42)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_val_groups = max(1, int(len(shuffled) * 0.2))
    val_groups = set(shuffled[:n_val_groups].tolist())
    train_pool_groups = shuffled[n_val_groups:]

    val_mask = np.isin(groups, list(val_groups))
    X_val, y_val = X[val_mask], y[val_mask]

    train_fractions = np.arange(0.1, 1.01, 0.1)
    train_sizes_groups, train_accs, val_accs = [], [], []

    for frac in train_fractions:
        n_groups_this_round = max(2, int(len(train_pool_groups) * frac))
        chosen_groups = set(train_pool_groups[:n_groups_this_round].tolist())
        mask = np.isin(groups, list(chosen_groups))
        X_sub, y_sub, groups_sub = X[mask], y[mask], groups[mask]
        if len(set(y_sub.tolist())) < 2:
            continue

        model = make_classifier()
        model.fit(X_sub, y_sub)
        y_train_pred = model.predict(X_sub)
        train_m = metrics_at_group_level(y_sub, y_train_pred, groups_sub)
        y_val_pred = model.predict(X_val)
        val_m = metrics_at_group_level(y_val, y_val_pred, groups[val_mask])
        train_accs.append(train_m["accuracy"])
        val_accs.append(val_m["accuracy"])
        train_sizes_groups.append(len(set(groups_sub.tolist())))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes_groups, train_accs, marker="o", color=TEAL, label="Training accuracy (group-level)")
    ax.plot(train_sizes_groups, val_accs, marker="o", color=LIGHT_BLUE, label="Validation accuracy "
                                                                            "(held-out genotypes, group-level)")
    ax.set_xlabel("Training set size (unique genotypes, cumulative % of pool)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Seed India Population-Only Model: Learning Curve\n"
                 "(validation genotypes never seen during training)")
    ax.set_ylim(0, 1.05)
    legend = ax.legend(facecolor=PANEL, edgecolor=GREY)
    for text in legend.get_texts():
        text.set_color(TEXT)
    ax.grid(alpha=0.15, color=GREY)
    _dark_axes(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=NAVY)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_feature_importances(out_path):
    model = load(os.path.join(ARTIFACTS_DIR, "india_calibrated_xgb.pkl"))
    importances = model.feature_importances_
    order = np.argsort(importances)
    names = np.array(CALIBRATION_CLASSIFIER_FEATURE_NAMES)[order]
    values = importances[order]

    highlight = {"india_diplotype_freq", "sas_vs_eur_ratio"}
    colors = [TEAL if n in highlight else GREY for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("India Population-Only Model: Feature Importances\n"
                 "(no CPIC score - SAS/EUR frequency structure)")
    _dark_axes(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=NAVY)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_confusion_matrix(records, out_path):
    y = np.array([
        population_adjusted_significance(r["clinical_significance"], r["sas_vs_eur_ratio"])
        for r in records
    ])
    groups = np.array(build_group_keys(records))
    X = np.array([build_calibration_classifier_features_from_entry(r) for r in records], dtype=float)

    y_pred, n_splits = grouped_cv_predict(X, y, groups, make_classifier, use_class_weights=True)

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Seed India Population-Only Model\nNormalised Confusion Matrix "
                 f"({n_splits}-fold grouped CV, adjusted significance label)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else TEXT, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT)
    _dark_axes(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=NAVY)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cyp2c19_population_comparison(out_path):
    df = pd.read_csv(CSV_PATH)

    def allele_freq(pop_df, gene, star):
        g = pop_df[pop_df["Gene"] == gene]
        total_alleles = len(g) * 2
        if total_alleles == 0:
            return 0.0
        count = sum(gt.split("/").count(star) for gt in g["Genotype"])
        return count / total_alleles

    freqs = {}
    for sup, label in [("SAS", "South Asian"), ("EUR", "European"), ("EAS", "East Asian")]:
        pop = df[df["Superpopulation code"] == sup]
        freqs[label] = allele_freq(pop, "CYP2C19", "*2")

    colors = {"South Asian": TEAL, "European": GREY, "East Asian": LIGHT_BLUE}

    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = list(freqs.keys())
    values = [freqs[k] * 100 for k in labels]
    bars = ax.bar(labels, values, color=[colors[k] for k in labels])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", color=TEXT)
    ax.set_ylabel("CYP2C19*2 allele frequency (%)")
    ax.set_title("CYP2C19*2 Allele Frequency by Population\n"
                 "(real 1000 Genomes Project data, live-computed)")
    ax.set_ylim(0, max(values) * 1.3)
    _dark_axes(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=NAVY)
    plt.close(fig)
    print(f"Saved {out_path}")
    return freqs


def main():
    with open(DATA_PATH) as f:
        records = dedupe_records_by_group(json.load(f))
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    plot_learning_curve(records, os.path.join(ARTIFACTS_DIR, "learning_curve.png"))
    plot_feature_importances(os.path.join(ARTIFACTS_DIR, "feature_importances.png"))
    plot_confusion_matrix(records, os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))
    freqs = plot_cyp2c19_population_comparison(
        os.path.join(ARTIFACTS_DIR, "cyp2c19_population_comparison.png"))
    print(f"\nCYP2C19*2 allele frequencies used in chart: "
          f"{ {k: round(v * 100, 1) for k, v in freqs.items()} }")


if __name__ == "__main__":
    main()
