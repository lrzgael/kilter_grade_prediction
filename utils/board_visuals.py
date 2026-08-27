#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualisation helpers for Kilter Board climbs.

Draws a climb (list of [hold_id, x, y, role]) on top of the board picture,
circling every used hold with a role-dependent colour:

    start  -> green
    hand   -> blue
    finish -> purple/magenta
    foot   -> orange

Usage (as a module):
    from utils.board_visuals import plot_climb
    plot_climb(holds, title="title")

"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.image as mpimg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMG = PROJECT_ROOT / "data" / "img" / "16x12SuperWide.png"

ROLE_MAP = {12: "start", 13: "hand", 14: "finish", 15: "foot"}

BOARD_EDGES_16x12 = {
    "edge_left": -24,
    "edge_right": 168,
    "edge_bottom": 0,
    "edge_top": 156,
}


ROLE_STYLE = {
    "start":  {"color": "#00c000", "lw": 2.6},   # green
    "hand":   {"color": "#1f77ff", "lw": 2.6},   # blue
    "finish": {"color": "#c000c0", "lw": 2.6},   # magenta
    "foot":   {"color": "#ff8c00", "lw": 2.2},   # orange
}
DEFAULT_STYLE = {"color": "red", "lw": 2.0}

# Draw order so that small foot circles do not hide hand circles
ROLE_ZORDER = {"foot": 2, "hand": 3, "start": 4, "finish": 5}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalise_role(role) -> str:
    """Accept either a numeric role id (12..15) or an already-mapped string."""
    if isinstance(role, str):
        return role.strip().lower()
    return ROLE_MAP.get(int(role), "unknown")


def parse_holds(holds) -> list[list]:
    """
    Accept a stringified list (as stored in data_cleaned.csv) or a real list
    of [hold_id, x, y, role]. Returns a list with roles as names.
    """
    if isinstance(holds, str):
        holds = ast.literal_eval(holds)
    parsed = []
    for h in holds:
        hold_id, x, y, role = h[0], h[1], h[2], h[3]
        if x is None or y is None:          # missing coordinates -> skip
            continue
        parsed.append([hold_id, float(x), float(y), _normalise_role(role)])
    return parsed


def _grid_to_pixel(x, y, img_w, img_h, edges=BOARD_EDGES_16x12):
    """
    Map board grid coordinates -> image pixel coordinates.

    The image spans exactly the board bounding box, its origin is top-left and
    its y axis points downwards, hence the vertical flip.
    """
    x0, x1 = edges["edge_left"], edges["edge_right"]
    y0, y1 = edges["edge_bottom"], edges["edge_top"]

    px = (x - x0) / (x1 - x0) * img_w
    py = (1.0 - (y - y0) / (y1 - y0)) * img_h     # flip
    return px, py


# --------------------------------------------------------------------------- #
# Main plotting function
# --------------------------------------------------------------------------- #

def plot_climb(
    holds,
    img_path: str | Path = DEFAULT_IMG,
    title: str | None = None,
    edges: dict = BOARD_EDGES_16x12,
    radius: float = 0.022,
    foot_radius_ratio: float = 0.72,
    ax=None,
    show_legend: bool = True,
    figsize: tuple = (9, 7.5),
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot a Kilter climb on the board picture.

    Parameters
    ----------
    holds : list | str
        List of [hold_id, x, y, role] (role as int 12..15 or as a name),
        or its stringified version as stored in data_cleaned.csv.
    img_path : path
        Board background image (default: data/img/16x12SuperWide.png).
    title : str, optional
        Figure title.
    edges : dict
        Board bounding box in grid units (default: 16x12 Original).
    radius : float
        Circle radius as a fraction of the image width.
    foot_radius_ratio : float
        Foot circles are drawn slightly smaller (they are small holds).
    ax : matplotlib Axes, optional
        Draw on an existing axes (useful for grids of climbs).
    show_legend, figsize, save_path, show :
        Usual cosmetics / IO switches.

    Returns
    -------
    matplotlib.axes.Axes
    """
    holds = parse_holds(holds)

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Board image not found: {img_path}\n"
            "Expected data/img/16x12SuperWide.png relative to the project root."
        )
    img = mpimg.imread(img_path)
    img_h, img_w = img.shape[0], img.shape[1]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(img, extent=(0, img_w, img_h, 0))   # keep pixel/top-left frame

    r_px = radius * img_w
    used_roles = []

    for hold_id, x, y, role in holds:
        px, py = _grid_to_pixel(x, y, img_w, img_h, edges)
        style = ROLE_STYLE.get(role, DEFAULT_STYLE)
        r = r_px * (foot_radius_ratio if role == "foot" else 1.0)

        ax.add_patch(
            mpatches.Circle(
                (px, py), r,
                facecolor="none",
                edgecolor=style["color"],
                linewidth=style["lw"],
                zorder=ROLE_ZORDER.get(role, 3),
            )
        )
        used_roles.append(role)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=11)

    if show_legend:
        order = ["start", "hand", "foot", "finish"]
        handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=10,
                   markerfacecolor="none",
                   markeredgecolor=ROLE_STYLE[r]["color"],
                   markeredgewidth=2.2, label=r)
            for r in order if r in set(used_roles)
        ]
        if handles:
            ax.legend(handles=handles, loc="upper right", frameon=True,
                      fontsize=9, framealpha=0.9)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show and ax.figure is not None:
        plt.show()

    return ax


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #

def _parse_args():
    p = argparse.ArgumentParser(description="Plot a Kilter climb on the board.")
    p.add_argument("--csv", type=str,
                   default=str(PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv"),
                   help="Cleaned CSV containing a 'holds' column.")
    p.add_argument("--row", type=int, default=0, help="Row index to plot.")
    p.add_argument("--img", type=str, default=str(DEFAULT_IMG))
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = _parse_args()
    df = pd.read_csv(args.csv)
    row = df.iloc[args.row]

    bits = []
    if "name" in row:
        bits.append(str(row["name"]))
    if "angle" in row:
        bits.append(f"{int(row['angle'])}°")
    if "difficulty_average" in row:
        bits.append(f"difficulty {row['difficulty_average']:.2f}")

    plot_climb(row["holds"], img_path=args.img,
               title="  |  ".join(bits), save_path=args.save)