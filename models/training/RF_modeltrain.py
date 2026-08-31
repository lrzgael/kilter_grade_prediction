#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the Kilter Board difficulty model (Random Forest).

Input : data/processed/data_processed_train.csv
        data/processed/data_processed_test.csv   (optional, only for quick sanity check)
Output: models/saved_models/RF_model.joblib

Usage:
    python models/training/RF_modeltrain.py --train-csv data/processed/data_processed_train.csv \
        --test-csv data/processed/data_processed_test.csv --model-out models/saved_models/RF_model.joblib
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

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
    """Train a random forest with sensible defaults (baseline)."""
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def tune_model(X_train, y_train, w_train, *, n_iter=20,
               random_state=DEFAULT_RANDOM_STATE, n_jobs=2):
    """Randomized hyper-parameter search, then refit the best config."""
    base = RandomForestRegressor(random_state=random_state)

    param_dist = {
        "n_estimators":      randint(100, 600),
        "max_depth":         randint(4, 40),
        "min_samples_split": randint(2, 50),
        "min_samples_leaf":  randint(1, 20),
        "max_features":      ["sqrt", "log2", 0.1, 0.3, 0.5],
        "max_samples":       [None, 0.6, 0.8],  # bootstrap subsampling
    }

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=n_jobs,
        pre_dispatch=4,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train, sample_weight=w_train)
    print(f"Best CV MAE: {-search.best_score_:.3f}")
    print("Best params:", search.best_params_)
    return search.best_estimator_


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", type=Path,
                        default="data/processed/data_processed_train.csv",
                        help="Path to the processed training CSV.")
    p.add_argument("--test-csv", type=Path,
                        default="data/processed/data_processed_test.csv",
                        help="Path to the processed test CSV (quick sanity check only).")
    p.add_argument("--model-out", type=Path,
                        default="models/saved_models/RF_model.joblib",
                        help="Where to save the trained model (.joblib).")
    p.add_argument("--n-iter", type=int, default=20,
                   help="Number of random search iterations")
    p.add_argument("--n-jobs", type=int, default=2,
                   help="Parallel jobs for the hyper-parameter search")
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return p.parse_args()


def main():
    args = parse_args()

    X_train, y_train, w_train = load_data(args.train_csv)

    # Tuning the model
    print("\nTuning RandomForestRegressor...")
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