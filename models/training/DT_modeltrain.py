#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the Kilter Board difficulty model (Decision Tree).

Input : data/processed/data_processed_train.csv
        data/processed/data_processed_test.csv   (optional, only for quick sanity check)
Output: models/saved_models/DT_model.joblib

Usage:
    python models/training/DT_modeltrain.py --train-csv data/processed/data_processed_train.csv \
        --test-csv data/processed/data_processed_test.csv --model-out models/saved_models/DT_model.joblib

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import loguniform, randint

TARGET = "difficulty_average"
NON_FEATURE_COLS = [TARGET, "ascensionist_count"]  # everything else is a feature
DEFAULT_RANDOM_STATE = 42  # for reproducibility in training


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def make_sample_weights(df, max_ascent_cap=100):
    """
    Weight is proportional to sqrt(ascensionist_count), capped.
    sqrt: reliability of an average grows with sqrt(n) (standard error!).
    """
    w = np.sqrt(df["ascensionist_count"].clip(upper=max_ascent_cap))
    w = w / w.min()                    # floor climb (3 ascents) -> weight 1.0
    return w.to_numpy(dtype=np.float32)


def load_data(path, verbose=True):
    """Load CSV and split into X / y / sample_weights."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in {path}. "
            f"Got: {list(df.columns)[:10]}..."
        )

    y = df[TARGET].astype(float)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # Safety: coerce everything to numeric (multihot cols may load as object)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if verbose:
        print(f"\nLoaded {path}")
        print(f"  {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
              f"range=[{y.min():.0f}, {y.max():.0f}]")

    sample_weights = make_sample_weights(df)
    return X, y, sample_weights


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(X_train, y_train, w_train, random_state=DEFAULT_RANDOM_STATE):
    """Train a decision tree with light leaf regularization (baseline)."""
    model = DecisionTreeRegressor(
        min_samples_leaf=20,
        random_state=random_state,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def tune_model(X_train, y_train, w_train, *, groups=None, n_iter=20,
               random_state=DEFAULT_RANDOM_STATE, n_jobs=2):
    """Randomized hyper-parameter search, then refit the best config."""
    base = DecisionTreeRegressor(random_state=random_state)

    param_dist = {
        "max_depth":        randint(2, 64),
        "min_samples_split": randint(2, 200),
        "min_samples_leaf":  randint(1, 100),
        "max_features":      [None, "sqrt", "log2", 0.5, 0.7],
        "ccp_alpha":         loguniform(1e-5, 1e-1),
    }

    cv = GroupKFold(n_splits=3) if groups is not None else 3

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=n_jobs,
        pre_dispatch=4,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train, sample_weight=w_train, groups=groups)

    print(f"\nBest CV MAE: {-search.best_score_:.3f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # No early stopping for a single tree; refit the best config on all data.
    final_model = DecisionTreeRegressor(
        **search.best_params_, random_state=random_state
    )
    final_model.fit(X_train, y_train, sample_weight=w_train)
    return final_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Decision Tree Kilter grade prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", type=Path,
                        default="data/processed/data_processed_train.csv",
                        help="Path to the processed training CSV.")
    parser.add_argument("--test-csv", type=Path,
                        default="data/processed/data_processed_test.csv",
                        help="Path to the processed test CSV (quick sanity check only).")
    parser.add_argument("--model-out", type=Path,
                        default="models/saved_models/DT_model.joblib",
                        help="Where to save the trained model (.joblib).")
    parser.add_argument("--random-state", type=int,
                        default=DEFAULT_RANDOM_STATE,
                        help="Random seed.")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of sampled parameter settings when --tune is used.")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="Parallel jobs for the hyper-parameter search.")
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, w_train = load_data(args.train_csv)


    # # Train model (only for tests) 
    # print("\nTraining DecisionTreeRegressor...")
    # model = train_model(X_train, y_train, w_train,
    #                     random_state=args.random_state)
        
        
    # Tune model
    print("\nTuning DecisionTreeRegressor...")
    model = tune_model(X_train, y_train, w_train,
                       n_iter=args.n_iter,
                       random_state=args.random_state,
                       n_jobs=args.n_jobs)

    # Persist model + the exact feature order it was trained on
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(X_train.columns), "target": TARGET},
        args.model_out,
    )
    print(f"\nSaved model -> {args.model_out}")

    # Quick sanity check (full evaluation lives in models/evaluation/model_evaluation.py)
    if args.test_csv is not None and Path(args.test_csv).exists():
        from sklearn.metrics import mean_absolute_error
        X_test, y_test, _ = load_data(args.test_csv, verbose=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        print(f"Quick check - Test MAE: "
              f"{mean_absolute_error(y_test, model.predict(X_test)):.3f} grades")
        print("Run models/evaluation/model_evaluation.py for the full report.")


if __name__ == "__main__":
    main()