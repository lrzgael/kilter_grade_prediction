#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation & feature-importance analysis for Kilter grade prediction models.

Input : models/saved_models/{MODEL}.joblib
        data/processed/data_processed_train.csv
        data/processed/data_processed_test.csv
Output: metrics printed to stdout
        figures saved in models/results/

Usage:
    python models/evaluation/model_evaluation.py --model models/saved_models/XGB_model.joblib

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")            # safe for headless runs; overridden by --show
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

TARGET = "difficulty_average"
NON_FEATURE_COLS = [TARGET, "ascensionist_count"]  # everything else is a predictive feature
DEFAULT_RANDOM_STATE = 42 # for reproducibility in compute feature importance

# French boulder grade (Font)
DIFFICULTY_TO_GRADE = {
    10: "4a",  11: "4b",  12: "4c",  13: "5a",  14: "5b",  15: "5c",
    16: "6a",  17: "6a+", 18: "6b",  19: "6b+", 20: "6c",  21: "6c+",
    22: "7a",  23: "7a+", 24: "7b",  25: "7b+", 26: "7c",  27: "7c+",
    28: "8a",  29: "8a+", 30: "8b",  31: "8b+", 32: "8c",  33: "8c+",
    34: "9a",  35: "9a+", 36: "9b",  37: "9b+", 38: "9c",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_dataset(path):
    """Load a processed dataset and return X, y."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}.")
    y = df[TARGET].astype(float)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y


def load_model(path):
    """
    Load a model saved by the training scripts.
    """
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("features")
    return obj, getattr(obj, "feature_names_in_", None)


def align_features(X, features):
    """Reorder / complete columns so they match the training feature set."""
    if features is None:
        return X
    return X.reindex(columns=list(features), fill_value=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred):
    """Regression + grade-tolerance metrics as a dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred_r = np.clip(np.round(y_pred), y_true.min(), y_true.max())
    y_true_r = np.round(y_true)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "exact": (y_pred_r == y_true_r).mean(),
        "within_1": (np.abs(y_pred_r - y_true_r) <= 1).mean(),
        "within_2": (np.abs(y_pred_r - y_true_r) <= 2).mean(),
    }


def print_metrics(m, label):
    print(f"\n===== {label} set evaluation =====")
    print(f"MAE            : {m['MAE']:.3f} grades")
    print(f"RMSE           : {m['RMSE']:.3f} grades")
    print(f"R2             : {m['R2']:.3f}")
    print(f"Exact grade    : {m['exact']:.1%}")
    print(f"Within +/-1    : {m['within_1']:.1%}")
    print(f"Within +/-2    : {m['within_2']:.1%}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def grade_ticks(vmin, vmax, step=1):
    """Return (tick_positions, tick_labels) for the grade axis."""
    lo, hi = int(np.floor(vmin)), int(np.ceil(vmax))
    ticks = list(range(lo, hi + 1, step))
    labels = [DIFFICULTY_TO_GRADE.get(t, str(t)) for t in ticks]
    return ticks, labels


def plot_pred_vs_true(y_true, y_pred, out_path, title=None, subtitle=None,
                      bins=20, show=False):
    """
    Seaborn jointplot: 2D histogram of (predicted, true) grades
    + marginal histograms + identity line, with grade tick labels.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip predictions to the range of true values 
    y_pred = np.clip(y_pred, y_true.min(), y_true.max())

    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5

    sns.set_theme(style="white", font_scale=1.0)

    g = sns.jointplot(
        x=y_pred, y=y_true,
        kind="hist", bins=bins, color="#4C86B0",
        height=6, ratio=5, space=0.05,
        marginal_kws=dict(bins=bins, edgecolor="black", linewidth=0.6),
        joint_kws=dict(cmap="Blues"),
    )

    g.ax_joint.plot([lo, hi], [lo, hi], color="#6A79E8", lw=1.5, zorder=5)
    g.ax_joint.set_xlim(lo, hi)
    g.ax_joint.set_ylim(lo, hi)

    ticks, labels = grade_ticks(lo, hi, step=1)
    g.ax_joint.set_xticks(ticks)
    g.ax_joint.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    g.ax_joint.set_yticks(ticks)
    g.ax_joint.set_yticklabels(labels, fontsize=8)

    g.ax_joint.set_xlabel("Predicted Grade")
    g.ax_joint.set_ylabel("True Grade")

    if title and subtitle:
        g.figure.suptitle(f"{title}\n{subtitle}", y=1.06)
    elif title:
        g.figure.suptitle(title, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")

    if show:
        plt.show()
    plt.close(g.figure)
    return g



def plot_feature_importance(importances, out_path, top_n=20, title=None, show=False):
    """Horizontal bar plot of the top-n permutation importances."""
    imp = importances.sort_values(ascending=False).head(top_n)[::-1]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 1.5))
    ax.barh(imp.index, imp.values, color="#4C86B0")
    ax.set_xlabel("Permutation importance ")
    if title:
        ax.set_title(title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(model, X, y, *, label, model_name, results_dir, plot=True, show=False):
    """Print metrics and save the prediction / residual figures."""
    y_pred = model.predict(X)
    m = compute_metrics(y, y_pred)
    print_metrics(m, label)

    if plot:
        subtitle = (f"MAE={m['MAE']:.2f}, within +/-1={m['within_1']:.0%}, "
                    f"within +/-2={m['within_2']:.0%}")
        plot_pred_vs_true(
            y, y_pred,
            out_path=results_dir / f"{model_name}_{label.lower()}.png",
            title=f"{model_name} - {label} dataset",
            subtitle=subtitle,
            show=show,
        )


    return y_pred, m


def compute_feature_importance(model, X, y, *, model_name, results_dir,
                               n_repeats=10, top_n=20,
                               random_state=DEFAULT_RANDOM_STATE, show=False):
    """Permutation importance on the test set + bar plot + CSV export."""
    print(f"\nComputing permutation importance ({n_repeats} repeats)...")
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    imp = pd.Series(result.importances_mean, index=X.columns)
    imp_std = pd.Series(result.importances_std, index=X.columns)

    print(f"\nTop {top_n} features:")
    print(imp.sort_values(ascending=False).head(top_n).to_string())

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{model_name}_feature_importance.csv"
    pd.DataFrame({"importance_mean": imp, "importance_std": imp_std}) \
        .sort_values("importance_mean", ascending=False) \
        .to_csv(csv_path, index_label="feature")
    print(f"Saved importances -> {csv_path}")

    plot_feature_importance(
        imp,
        out_path=results_dir / f"{model_name}_feature_importance.png",
        top_n=top_n,
        title=f"{model_name} - top {top_n} features (permutation importance)",
        show=show,
    )
    return imp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Kilter grade prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, 
                        default="models/saved_models/RF_model.joblib",
                        help="Path to the trained .joblib model.")
    parser.add_argument("--train-csv", type=Path, 
                        default="data/processed/data_processed_train.csv",
                        help="Path to the processed training CSV.")
    parser.add_argument("--test-csv", type=Path, 
                        default="data/processed/data_processed_test.csv",
                        help="Path to the processed test CSV.")
    parser.add_argument("--results-dir", type=Path, 
                        default="models/results/",
                        help="Directory where figures and CSVs are saved.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Random seed (permutation importance).")
    parser.add_argument("--no-train-eval", action="store_true",
                        help="Skip evaluation on the training set.")
    parser.add_argument("--importance", action="store_true",
                        help="Compute permutation feature importance (slow).")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Repeats for permutation importance.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of features shown in the importance plot.")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively in addition to saving them.")
    return parser.parse_args()


def main(feature_importances = True):
    args = parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    model, features = load_model(args.model)
    model_name = args.model.stem
    print(f"Loaded model '{model_name}' from {args.model}")

    # Evaluation on both test and train datasets
    X_test, y_test = load_dataset(args.test_csv)
    X_test = align_features(X_test, features)
    evaluate(model, X_test, y_test, label="Test", model_name=model_name,
             results_dir=args.results_dir, show=args.show)

    X_train, y_train = load_dataset(args.train_csv)
    X_train = align_features(X_train, features)
    evaluate(model, X_train, y_train, label="Train", model_name=model_name,
             results_dir=args.results_dir, show=args.show)
        

    # compute features importance of the model
    if feature_importances :
        compute_feature_importance(
            model, X_test, y_test,
            model_name=model_name,
            results_dir=args.results_dir,
            n_repeats=args.n_repeats,
            top_n=args.top_n,
            random_state=args.random_state,
            show=args.show,
        )
        


if __name__ == "__main__":
    main()