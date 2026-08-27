#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Builds ML-ready features from the extracted Kilter Board climb dataset:
  1. Multi-hot encoding of holds (by role: start/hand/foot/finish)
  2. Hand-crafted geometric/statistical features

Outputs two files (train/test split) 

Usage : python utils/feature_engineering.py  --input_csv data/processed/data_cleaned.csv \
    --output_test_csv data/processed/data_processed_test.csv \
    --output_train_csv data/processed/data_processed_train.csv

"""

import ast
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Map numeric role IDs to names: 12=start, 13=hand, 14=finish, 15=foot
ROLE_MAP = {12: "start", 13: "hand", 14: "finish", 15: "foot"}
ROLES = list(ROLE_MAP.values())  # ["start", "hand", "finish", "foot"]


def parse_holds(holds_str):
    """holds_str: stringified list of [hold_id, x, y, role_id]."""
    if isinstance(holds_str, str):
        holds_str = ast.literal_eval(holds_str)
    # Convert numeric role -> name once here, rest of pipeline unchanged
    return [[h[0], h[1], h[2], ROLE_MAP[h[3]]] for h in holds_str]


# ---------- Multi-hot encoding ----------

def build_multihot_features(df: pd.DataFrame, min_freq: int = 5) -> pd.DataFrame:
    """One column per (hold_id, role) combo used at least `min_freq` times."""
    df["holds"] = df["holds"].apply(parse_holds)

    # Count frequency of each (hold_id, role) across dataset
    counts = {}
    for holds in df["holds"]:
        for hold in holds:
            hold_id, x, y, role = hold
            key = (hold_id, role)
            counts[key] = counts.get(key, 0) + 1

    valid_keys = [k for k, v in counts.items() if v >= min_freq]
    key_to_col = {k: f"hold_{k[0]}_{k[1]}" for k in valid_keys}

    multihot = pd.DataFrame(0, index=df.index, columns=list(key_to_col.values()))
    for idx, holds in zip(df.index, df["holds"]):
        for hold in holds:
            hold_id, x, y, role = hold
            key = (hold_id, role)
            if key in key_to_col:
                multihot.at[idx, key_to_col[key]] = 1

    return multihot


# ---------- Geometric features ----------

def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hand-crafted spatial/statistical descriptors of each climb.

    """
    records = []
    for holds, angle in zip(df["holds"], df["angle"]):
        by_role = {r: [(x, y) for _, x, y, role in holds if role == r] for r in ROLES}
        all_pts = [(x, y) for _, x, y, _ in holds]

        # Hand-grabbed holds: hand + start + finish (all touched by hand)
        hand_pts = by_role["hand"] + by_role["start"] + by_role["finish"]
        hand_sorted = sorted(hand_pts, key=lambda p: p[1])  # bottom to top (proxy order)

        xs, ys = zip(*all_pts) if all_pts else ([0], [0])
        board_width = max(xs) - min(xs) if all_pts else 0
        board_height = max(ys) - min(ys) if all_pts else 0

        # All pairwise distances between hand holds (global spread measure)
        dists = (
            [euclidean(a, b) for a, b in itertools.combinations(hand_pts, 2)]
            if len(hand_pts) > 1 else [0]
        )
        
        # All pairwise distances between foot holds
        foot_dists = [euclidean(a,b) for a,b in itertools.combinations(by_role["foot"], 2)] or [0]


        # Move distances in climbing order 
        # proxy ordering : only consecutive ones along the vertical ordering
        consecutive_dists = (
            [euclidean(a, b) for a, b in zip(hand_sorted, hand_sorted[1:])]
            or [0]
        )


        records.append({
            # --- counts ---
            "n_hand": len(by_role["hand"]),
            "n_foot": len(by_role["foot"]),
            "n_holds_total": len(all_pts),

            # --- board geometry ---
            "board_width": board_width,
            "board_height": board_height,
            "hold_density": len(all_pts) / max(board_width * board_height, 1),

            # --- hand holds distances (global spread) ---
            "avg_move_dist": np.mean(dists),
            "max_move_dist": np.max(dists),
            "std_move_dist": np.std(dists),
            
            # --- foot holds ---
            "avg_foot_dist": np.mean(foot_dists),
            "max_foot_dist": np.max(foot_dists),
            "std_foot_dist": np.std(foot_dists),

            # --- foot/hand relationship ---
            "foot_to_hand_ratio": len(by_role["foot"]) / max(len(hand_pts), 1),
            "hand_foot_dist": (
                euclidean(
                    (np.mean([p[0] for p in hand_pts]), np.mean([p[1] for p in hand_pts])),
                    (np.mean([p[0] for p in by_role["foot"]]), np.mean([p[1] for p in by_role["foot"]])),
                )
            ) if by_role["foot"] else 0,

            # --- consecutive (proxy-ordered) move distances ---
            "avg_consecutive_move": np.mean(consecutive_dists),
            "max_consecutive_move": np.max(consecutive_dists),  # crux proxy
            "std_consecutive_move": np.std(consecutive_dists),


            # --- overall climb span ---
            "start_finish_dist": (
                euclidean(
                    (np.mean([p[0] for p in by_role["start"]]), np.mean([p[0] for p in by_role["start"]])),
                    (np.mean([p[0] for p in by_role["finish"]]), np.mean([p[1] for p in by_role["finish"]])),
                )
            ) if by_role["finish"] and by_role["start"] else 0,
            
            
            # Interactions with angle: 
            "move_dist_x_angle": np.max(consecutive_dists) * angle, #crux move distance scaled by wall angle.
            "avg_consecutive_move_x_angle": np.mean(consecutive_dists) * angle,
            "max_foot_dist_x_angle": np.max(foot_dists) * angle,
        })

    return pd.DataFrame(records, index=df.index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for Kilter Board climb data."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../data/processed/data_cleaned.csv",
        help="Path to cleaned input CSV.",
    )
    parser.add_argument(
        "--output_train_csv",
        type=str,
        default="../data/processed/data_processed_train.csv",
        help="Path to output training CSV.",
    )
    parser.add_argument(
        "--output_test_csv",
        type=str,
        default="../data/processed/data_processed_test.csv",
        help="Path to output test CSV.",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=10,
        help="Minimum frequency for a (hold_id, role) combo to get its own multi-hot column.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test split (default: 0.1).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    multihot_feats = build_multihot_features(df, min_freq=args.min_freq)
    geo_feats = build_geometric_features(df)

    result = pd.concat(
        [df[["difficulty_average", "ascensionist_count", "angle", "n_holds", "is_nomatch"]],
         geo_feats, multihot_feats],
        axis=1,
    )

    train_df, test_df = train_test_split(
        result,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df.to_csv(args.output_train_csv, index=False)
    test_df.to_csv(args.output_test_csv, index=False)

    print(f"Saved {train_df.shape[0]} rows x {train_df.shape[1]} features to {args.output_train_csv}")
    print(f"Saved {test_df.shape[0]} rows x {test_df.shape[1]} features to {args.output_test_csv}")