#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Extracts climb + difficulty data from a BoardLib SQLite DB (Kilter Board)
and structures it into a clean pandas DataFrame ready for ML.

Usage : python utils/data_preparation.py --db_input data/raw/raw_kilter_data.db --csv_output data/processed/data_cleaned.csv

"""

import sqlite3
import pandas as pd
import argparse

TARGET_LAYOUT_NAME = "Kilter Board Original"
TARGET_BOARD_SIZE = "16x12"

VALID_ROLES = {12, 13, 14, 15}  # roles for KilterBoard original :start, hand, finish, foot


# ---------- Grade mapping (BoardLib difficulty_grades table) ----------
# display_difficulty (int) -> French boulder grade
DIFFICULTY_TO_GRADE = {
    10: "4a",  11: "4b",  12: "4c",  13: "5a",  14: "5b",  15: "5c",
    16: "6a",  17: "6a+", 18: "6b",  19: "6b+", 20: "6c",  21: "6c+",
    22: "7a",  23: "7a+", 24: "7b",  25: "7b+", 26: "7c",  27: "7c+",
    28: "8a",  29: "8a+", 30: "8b",  31: "8b+", 32: "8c",  33: "8c+",
    34: "9a",  35: "9a+", 36: "9b",  37: "9b+", 38: "9c",
}

DIFF_MIN = 15 #6a
DIFF_MAX = 27 #7c
ASCEND_MIN = 5
QUAL_MIN = 2


def get_layout_id(conn: sqlite3.Connection, layout_name: str) -> int:
    """Look up the layout_id for a given layout name (robust to ID changes)."""
    layouts = pd.read_sql_query("SELECT id, name FROM layouts", conn)
    match = layouts[layouts["name"].str.strip().str.lower() == layout_name.lower()]
    if match.empty:
        available = ", ".join(layouts["name"].tolist())
        raise ValueError(f"Layout '{layout_name}' not found. Available: {available}")
    return int(match.iloc[0]["id"])

def get_board_edges(board_size: str):
    
    match board_size:
        case "16x12":
            board_edges = {"edge_left": -24, "edge_right":168,
                           "edge_bottom": 0, "edge_top":156}
        case "12x12":
            board_edges = {"edge_left": 0, "edge_right":144,
                           "edge_bottom": 0, "edge_top":156}
        case "8x12":
            board_edges = {"edge_left": 24, "edge_right":120,
                           "edge_bottom": 0, "edge_top":156}
        case "7x10":
            board_edges = {"edge_left": 28, "edge_right":116,
                           "edge_bottom": 36, "edge_top":156}
        
    return board_edges

def load_raw_tables(db_path: str, layout_id: int, board_size : str, 
                    DIFF_MIN=DIFF_MIN, DIFF_MAX = DIFF_MAX, 
                    ASCEND_MIN = ASCEND_MIN, QUAL_MIN = QUAL_MIN):
    """Load relevant tables from the BoardLib SQLite database, filtered by layout."""
    conn = sqlite3.connect(db_path)

    board_edges = get_board_edges(board_size)

    climbs = pd.read_sql_query(
    """
    SELECT uuid, name, frames, angle, layout_id, is_draft,
           edge_left, edge_right, edge_bottom, edge_top, is_nomatch
    FROM climbs
    WHERE is_draft = 0
          AND layout_id = ?
          AND edge_left > ?
          AND edge_right < ?
          AND edge_bottom > ?
          AND edge_top < ?
    """,
    conn,
    params=(layout_id,board_edges["edge_left"],board_edges["edge_right"],
            board_edges["edge_bottom"],board_edges["edge_top"],),
    )
    stats = pd.read_sql_query(
        """
        SELECT climb_uuid, angle, display_difficulty,
               difficulty_average, quality_average, ascensionist_count
        FROM climb_stats
        WHERE ascensionist_count >= ?
              AND quality_average >= ?
              AND difficulty_average BETWEEN ? AND ?
        """,
        conn,
        params=(ASCEND_MIN, QUAL_MIN, DIFF_MIN, DIFF_MAX,)
    )

    conn.close()
    return climbs, stats

def merge_climbs_and_stats(climbs: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Merge climb geometry with climbs stats."""
    df = stats.merge(
        climbs, left_on=["climb_uuid", "angle"], right_on=["uuid", "angle"], how="inner"
    )
    df = df.drop(columns=["uuid"])
    return df

def parse_frames(frames_str: str) -> list[tuple[int, int]]:
    """
    Parse 'frames' (e.g. 'p1234r12p1235r13...') into
    (placement_id, role_id) tuples.
    """
    holds = []
    if not isinstance(frames_str, str):
        return holds
    parts = frames_str.replace("p", " p").split()
    for part in parts:
        if not part.startswith("p"):
            continue
        try:
            placement_id, role_id = part[1:].split("r")
            holds.append((int(placement_id), int(role_id)))
        except ValueError:
            continue
    return holds

def get_placement_coordinates(
    conn: sqlite3.Connection, placement_ids: list[int]
) -> dict[int, tuple[float, float]]:
    """
    Resolve placement_id -> hold_id (placements table) -> x, y (holes table)
    in a single JOIN. Returns {placement_id: (x, y)}.
    """
    if not placement_ids:
        return {}

    unique_ids = list(set(placement_ids))
    placeholders = ",".join(["?"] * len(unique_ids))

    query = f"""
    SELECT p.id AS placement_id, h.x, h.y
    FROM placements p
    JOIN holes h ON h.id = p.hole_id
    WHERE p.id IN ({placeholders})
    """

    try:
        coords_df = pd.read_sql_query(query, conn, params=unique_ids)
        coords = dict(
            zip(coords_df["placement_id"], zip(coords_df["x"], coords_df["y"]))
        )

        missing = set(unique_ids) - set(coords_df["placement_id"])
        if missing:
            print(f"Warning: {len(missing)} placements have no coordinates "
                  f"(e.g. {sorted(missing)[:5]})")
        return coords

    except Exception as e:
        print(f"Error fetching placement coordinates: {e}")
        return {}

def build_hold_features(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Enrich each climb's holds with (x, y) coordinates.
    Input  'holds': [(placement_id, role_id), ...]
    Output 'holds': [(placement_id, x, y, role_id), ...]
    """
    df["holds"] = df["frames"].apply(parse_frames)# Collect all unique placement_ids across the dataset
    all_placement_ids = list({pid for holds in df["holds"] for pid, _ in holds})

    placement_coords = get_placement_coordinates(conn, all_placement_ids)

    # Enrich with coordinates
    def enrich(holds):
        return [
            (pid, *placement_coords.get(pid, (None, None)), role_id)
            for pid, role_id in holds
        ]

    df["holds"] = df["holds"].apply(enrich)
    df["n_holds"] = df["holds"].apply(len)
    df["placement_ids"] = df["holds"].apply(lambda h: [x[0] for x in h])


    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, missing targets, and unrealistic climbs."""
    df = df.dropna(subset=["display_difficulty", "frames", "difficulty_average"])
    df = df.drop_duplicates(subset=["climb_uuid", "angle"])
    df = df[df["n_holds"] >= 3]

    # Drop climbs containing unknown role IDs (e.g. 20, 21, 23), layout = Original
    n_before = len(df)
    df = df[df["holds"].apply(
        lambda holds: all(h[3] in VALID_ROLES for h in holds)
    )]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Data cleaning : dropped {n_dropped} climbs with invalid role IDs")
    return df.reset_index(drop=True)

def main(db_input: str, csv_output: str) -> None:
    """Main function to extract and prepare Kilter Board data."""
    # Connect once and reuse
    conn = sqlite3.connect(db_input)

    try:
        layout_id = get_layout_id(conn, TARGET_LAYOUT_NAME)
        climbs, stats = load_raw_tables(db_input, layout_id, TARGET_BOARD_SIZE)

        # Merge data
        df = merge_climbs_and_stats(climbs, stats)

        # Build features with database connection
        df = build_hold_features(df, conn)  # Pass the connection here

        # Clean dataset
        df = clean_dataset(df)

        # Select final columns
        final_cols = [
            "climb_uuid", "name", "angle", "layout_id",
            "n_holds", "holds",
            "difficulty_average", "quality_average", "ascensionist_count", "is_nomatch"
        ]
        df = df[final_cols]

        df.to_csv(csv_output, index=False)
        print(f"Saved {len(df)} climbs (layout='{TARGET_LAYOUT_NAME}', board size={TARGET_BOARD_SIZE}, to {csv_output}")
    finally:
        # Ensure connection is closed even if an error occurs
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract climb and difficulty data from Kilter Board SQLite database."
    )
    parser.add_argument(
        "--db_input",
        type=str,
        default="data/raw/raw_kilter_data.db",
        help="Path to the BoardLib SQLite database file"
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="data/processed/data_cleaned.csv",
        help="Path to save the cleaned CSV output file"
    )

    args = parser.parse_args()
    main(db_input=args.db_input, csv_output=args.csv_output)