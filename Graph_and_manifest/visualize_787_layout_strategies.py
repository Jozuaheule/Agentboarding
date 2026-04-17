from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# -----------------------------
# Editable run settings
# -----------------------------
NODES_FILE = "nodes_787.xlsx"
EDGES_FILE = "edges_787.xlsx"
MANIFEST_FILE = "generated_manifest.xlsx"  # Optional. Used to mark occupied seats if desired.
OUTPUT_DIR = "boarding_group_maps"

SAVE_FIGURES = True
SHOW_FIGURES = True
SHOW_NODE_LABELS = False
X_AXIS_TICK_STEP = 2
COLOR_ONLY_MANIFEST_SEATS = False  # False -> color the full seat layout by group.

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input_path(filename: str) -> Path:
    return SCRIPT_DIR / filename


def resolve_output_path(filename: str) -> Path:
    return SCRIPT_DIR / OUTPUT_DIR / filename


# -----------------------------
# Static cabin definitions
# -----------------------------
BUSINESS_ROWS = range(1, 8)
PREMIUM_ROWS = range(8, 14)
ECONOMY_ROWS = range(14, 39)

BUSINESS_Y_TO_LETTER = {0: "A", 4: "D", 6: "F", 10: "K"}
PREMIUM_Y_TO_LETTER = {0: "A", 2: "C", 4: "D", 5: "E", 6: "F", 8: "H", 10: "K"}
ECONOMY_Y_TO_LETTER = {0: "A", 1: "B", 2: "C", 4: "D", 5: "E", 6: "F", 8: "G", 9: "H", 10: "K"}

BASE_NODE_COLORS = {
    "aisle": "lightgray",
    "galley": "gray",
}

NEUTRAL_SEAT_COLOR = "#d9e8fb"

STRATEGY_GROUPS = {
    "zone_std": {
        "column": "zone_std",
        "title": "B787 Cabin Layout - Standard Zonal Boarding",
        "filename": "strategy_zone_std.png",
        "labels": {
            1: "Group 1 - Business (rows 1-7)",
            2: "Group 2 - Premium Economy (rows 8-13)",
            3: "Group 3 - Economy Rear (rows 30-38)",
            4: "Group 4 - Economy Mid (rows 22-29)",
            5: "Group 5 - Economy Front (rows 14-21)",
        },
        "colors": {
            1: "#4c78a8",
            2: "#f58518",
            3: "#54a24b",
            4: "#e45756",
            5: "#72b7b2",
        },
    },
    "zone_pyramid": {
        "column": "zone_pyramid",
        "title": "B787 Cabin Layout - Pyramid Boarding",
        "filename": "strategy_zone_pyramid.png",
        "labels": {
            1: "Group 1 - Business",
            2: "Group 2 - Rear farthest",
            3: "Group 3 - Forward farthest + rear intermediate",
            4: "Group 4 - Rear aisle + forward intermediate",
            5: "Group 5 - Forward aisle",
        },
        "colors": {
            1: "#4c78a8",
            2: "#f58518",
            3: "#54a24b",
            4: "#e45756",
            5: "#b279a2",
        },
    },
}


def load_graph(nodes_file: str = NODES_FILE, edges_file: str = EDGES_FILE) -> nx.DiGraph:
    df_nodes = pd.read_excel(resolve_input_path(nodes_file))
    df_edges = pd.read_excel(resolve_input_path(edges_file))

    graph = nx.DiGraph()

    x_col = "x" if "x" in df_nodes.columns else "x_pos"
    y_col = "y" if "y" in df_nodes.columns else "y_pos"

    for _, row in df_nodes.iterrows():
        node_id = row["id"]
        node_type = "seat" if row["type"] == "stand" else row["type"]
        graph.add_node(
            node_id,
            x=int(row[x_col]),
            y=int(row[y_col]),
            type=node_type,
        )

    for _, row in df_edges.iterrows():
        graph.add_edge(row["from"], row["to"], length=row["length"])

    return graph


def load_manifest(manifest_file: str) -> Optional[pd.DataFrame]:
    manifest_path = resolve_input_path(manifest_file)
    if not manifest_path.exists():
        return None

    manifest_df = pd.read_excel(manifest_path)
    required_columns = {"x_coord", "y_coord"}
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Manifest file is missing columns: {sorted(missing_columns)}")

    manifest_df["x_coord"] = manifest_df["x_coord"].astype(int)
    manifest_df["y_coord"] = manifest_df["y_coord"].astype(int)
    return manifest_df


def get_class_from_row(row_number: int) -> str:
    if row_number in BUSINESS_ROWS:
        return "business"
    if row_number in PREMIUM_ROWS:
        return "premium_economy"
    if row_number in ECONOMY_ROWS:
        return "economy"
    raise ValueError(f"Unsupported row number: {row_number}")


def get_seat_letter(row_number: int, y_coord: int) -> str:
    if row_number in BUSINESS_ROWS:
        mapping = BUSINESS_Y_TO_LETTER
    elif row_number in PREMIUM_ROWS:
        mapping = PREMIUM_Y_TO_LETTER
    else:
        mapping = ECONOMY_Y_TO_LETTER

    if y_coord not in mapping:
        raise ValueError(
            f"Unexpected y-coordinate {y_coord} for row {row_number}. "
            f"Allowed values are {sorted(mapping.keys())}."
        )
    return mapping[y_coord]


def get_seat_depth_category(row_number: int, seat_letter: str) -> str:
    if row_number in BUSINESS_ROWS:
        return "business_direct"

    if row_number in PREMIUM_ROWS:
        if seat_letter in {"A", "E", "K"}:
            return "maximum"
        return "aisle"

    if seat_letter in {"A", "E", "K"}:
        return "maximum"
    if seat_letter in {"B", "D", "F", "H"}:
        return "intermediate"
    return "aisle"


def get_zone_std(row_number: int) -> int:
    if row_number in BUSINESS_ROWS:
        return 1
    if row_number in PREMIUM_ROWS:
        return 2
    if 30 <= row_number <= 38:
        return 3
    if 22 <= row_number <= 29:
        return 4
    if 14 <= row_number <= 21:
        return 5
    raise ValueError(f"Unsupported row number for Zone_STD: {row_number}")


def get_zone_pyramid(row_number: int, seat_letter: str) -> int:
    if row_number in BUSINESS_ROWS:
        return 1

    seat_depth_category = get_seat_depth_category(row_number, seat_letter)
    is_rear = 24 <= row_number <= 38
    is_forward = 8 <= row_number <= 23

    if is_rear and seat_depth_category == "maximum":
        return 2

    if (is_forward and seat_depth_category == "maximum") or (
        is_rear and seat_depth_category == "intermediate"
    ):
        return 3

    if (is_rear and seat_letter in {"C", "G"}) or (
        14 <= row_number <= 23 and seat_depth_category == "intermediate"
    ):
        return 4

    if is_forward:
        return 5

    raise ValueError(
        f"Could not assign Zone_PYRAMID for row {row_number}, seat letter {seat_letter}."
    )


def build_seat_dataframe(graph: nx.DiGraph) -> pd.DataFrame:
    seat_records = []

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "seat":
            continue
        seat_records.append({
            "node_id": node_id,
            "x_coord": int(data["x"]),
            "y_coord": int(data["y"]),
        })

    seat_df = pd.DataFrame(seat_records)
    row_x_values = sorted(seat_df["x_coord"].unique())
    x_to_row_number = {x_value: index + 1 for index, x_value in enumerate(row_x_values)}

    seat_df["row_number"] = seat_df["x_coord"].map(x_to_row_number)
    seat_df["class"] = seat_df["row_number"].map(get_class_from_row)
    seat_df["seat_letter"] = seat_df.apply(
        lambda row: get_seat_letter(int(row["row_number"]), int(row["y_coord"])),
        axis=1,
    )
    seat_df["seat_depth_category"] = seat_df.apply(
        lambda row: get_seat_depth_category(int(row["row_number"]), str(row["seat_letter"])),
        axis=1,
    )
    seat_df["group_front_rear"] = seat_df["row_number"].apply(lambda row_number: 1 if row_number <= 18 else 2)
    seat_df["zone_std"] = seat_df["row_number"].map(get_zone_std)
    seat_df["zone_pyramid"] = seat_df.apply(
        lambda row: get_zone_pyramid(int(row["row_number"]), str(row["seat_letter"])),
        axis=1,
    )

    return seat_df.sort_values(["row_number", "y_coord"]).reset_index(drop=True)


def attach_manifest_occupancy(seat_df: pd.DataFrame, manifest_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if manifest_df is None:
        seat_df["occupied"] = True
        return seat_df

    occupied_pairs = set(zip(manifest_df["x_coord"], manifest_df["y_coord"]))
    seat_df["occupied"] = seat_df.apply(
        lambda row: (int(row["x_coord"]), int(row["y_coord"])) in occupied_pairs,
        axis=1,
    )
    return seat_df


def style_axes(pos: Dict[str, tuple[float, float]]) -> None:
    plt.axis("equal")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(X_AXIS_TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    x_vals = [point[0] for point in pos.values()]
    y_vals = [point[1] for point in pos.values()]
    x_min, x_max = math.floor(min(x_vals)), math.ceil(max(x_vals))
    y_min, y_max = math.floor(min(y_vals)), math.ceil(max(y_vals))

    ax.set_xticks(range(x_min, x_max + 1, X_AXIS_TICK_STEP))
    ax.set_yticks(range(y_min, y_max + 1, 1))
    ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlabel("")
    plt.ylabel("")


def plot_base_graph(graph: nx.DiGraph) -> None:
    pos = {node_id: (data["x"], data["y"]) for node_id, data in graph.nodes(data=True)}

    plt.figure(figsize=(14, 6))
    nx.draw_networkx_edges(graph, pos, edge_color="darkgray", width=0.6, arrows=False)

    for node_type, color in {**BASE_NODE_COLORS, "seat": "blue"}.items():
        nodelist = [node_id for node_id, data in graph.nodes(data=True) if data.get("type") == node_type]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=28,
            label=node_type,
        )

    if SHOW_NODE_LABELS:
        labels = {node_id: str(node_id) for node_id in graph.nodes}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

    style_axes(pos)
    plt.title("B787-inspired Custom Cabin Layout - Network Graph")
    plt.legend(title="Node Type")
    plt.tight_layout()

    if SAVE_FIGURES:
        output_path = resolve_output_path("layout_base.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def build_strategy_legend(group_labels: Dict[int, str], group_colors: Dict[int, str]) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=group_colors[group_id], markersize=8)
        for group_id, label in group_labels.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="w", label="Aisle", markerfacecolor=BASE_NODE_COLORS["aisle"], markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Galley", markerfacecolor=BASE_NODE_COLORS["galley"], markersize=8),
        ]
    )

    if COLOR_ONLY_MANIFEST_SEATS:
        handles.append(
            Line2D([0], [0], marker="o", color="w", label="Seat not in manifest", markerfacecolor=NEUTRAL_SEAT_COLOR, markersize=8)
        )

    return handles


def plot_strategy_map(graph: nx.DiGraph, seat_df: pd.DataFrame, strategy_key: str) -> None:
    strategy = STRATEGY_GROUPS[strategy_key]
    group_column = strategy["column"]
    group_labels = strategy["labels"]
    group_colors = strategy["colors"]

    pos = {node_id: (data["x"], data["y"]) for node_id, data in graph.nodes(data=True)}

    plt.figure(figsize=(14, 6))
    nx.draw_networkx_edges(graph, pos, edge_color="darkgray", width=0.6, arrows=False)

    for node_type, color in BASE_NODE_COLORS.items():
        nodelist = [node_id for node_id, data in graph.nodes(data=True) if data.get("type") == node_type]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=28,
        )

    if COLOR_ONLY_MANIFEST_SEATS:
        unoccupied_nodes = seat_df.loc[~seat_df["occupied"], "node_id"].tolist()
        if unoccupied_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=unoccupied_nodes,
                node_color=NEUTRAL_SEAT_COLOR,
                node_size=28,
            )
        colored_df = seat_df.loc[seat_df["occupied"]].copy()
    else:
        colored_df = seat_df.copy()

    for group_id, color in group_colors.items():
        nodelist = colored_df.loc[colored_df[group_column] == group_id, "node_id"].tolist()
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=36,
        )

    if SHOW_NODE_LABELS:
        labels = {node_id: str(node_id) for node_id in graph.nodes}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

    style_axes(pos)
    plt.title(strategy["title"])
    plt.legend(
        handles=build_strategy_legend(group_labels, group_colors),
        title="Boarding groups",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    plt.tight_layout()

    if SAVE_FIGURES:
        output_path = resolve_output_path(strategy["filename"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def main() -> None:
    graph = load_graph(NODES_FILE, EDGES_FILE)
    manifest_df = load_manifest(MANIFEST_FILE)
    seat_df = build_seat_dataframe(graph)
    seat_df = attach_manifest_occupancy(seat_df, manifest_df)

    plot_base_graph(graph)
    plot_strategy_map(graph, seat_df, "front_rear")
    plot_strategy_map(graph, seat_df, "zone_std")
    plot_strategy_map(graph, seat_df, "zone_pyramid")


if __name__ == "__main__":
    main()
