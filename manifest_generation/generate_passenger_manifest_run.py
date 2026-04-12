
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from calibration.calibration_config import (
    CALIBRATION_LOAD_FACTOR,
    CALIBRATION_LUGGAGE_PROBABILITY,
    CALIBRATION_N_MANIFESTS,
    STOW_DIST,
    STOW_SCALE_S,
    STOW_SHAPE,
    STOW_UNIFORM_HIGH_S,
    STOW_UNIFORM_LOW_S,
    StowConfig,
    validate_bounds,
)


# ============================================================
# USER SETTINGS
# Edit these values and run the script directly with Play.
# The script resolves files in this order:
#   manifest_generation/graph/
# then local fallback locations for compatibility.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

NODES_FILE = "nodes_787.xlsx"
EDGES_FILE = "edges_787.xlsx"
OUTPUT_FILE = "generated_manifest.xlsx"  # .xlsx or .csv

LOAD_FACTOR = CALIBRATION_LOAD_FACTOR
BUSINESS_LOAD_FACTOR: Optional[float] = None
PREMIUM_LOAD_FACTOR: Optional[float] = None
ECONOMY_LOAD_FACTOR: Optional[float] = None

LUGGAGE_PROBABILITY = CALIBRATION_LUGGAGE_PROBABILITY

SEED: Optional[int] = 42

# Number of unique manifest files to simulate
N_MANIFESTS = CALIBRATION_N_MANIFESTS

DEFAULT_STOW_CONFIG = StowConfig(
    dist=STOW_DIST,
    shape_k=STOW_SHAPE,
    scale_lambda_s=STOW_SCALE_S,
    low_s=STOW_UNIFORM_LOW_S,
    high_s=STOW_UNIFORM_HIGH_S,
)


# -----------------------------
# Static cabin definitions
# -----------------------------
BUSINESS_ROWS = range(1, 8)
PREMIUM_ROWS = range(8, 14)
ECONOMY_ROWS = range(14, 39)

BUSINESS_Y_TO_LETTER = {0: "A", 4: "D", 6: "F", 10: "K"}
PREMIUM_Y_TO_LETTER = {0: "A", 2: "C", 4: "D", 5: "E", 6: "F", 8: "H", 10: "K"}
ECONOMY_Y_TO_LETTER = {0: "A", 1: "B", 2: "C", 4: "D", 5: "E", 6: "F", 8: "G", 9: "H", 10: "K"}

REQUIRED_NODE_COLUMNS = {"id", "x", "y", "type"}
REQUIRED_EDGE_COLUMNS = {"from", "to", "length"}

OUTPUT_COLUMNS = [
    "pax_id",
    "x_coord",
    "y_coord",
    "seat",
    "class",
    "preferred_door",
    "preferred_spawn",
    "alternative_spawn",
    "assigned_aisle",
    "preferred_speed",
    "lateral_speed",
    "has_luggage",
    "stow_duration",
    "zone_std",
    "zone_outsidein",
    "zone_pyramid",
]

REAR_OUTER_AISLE_ECONOMY_SEATS = {"C", "G"}


def validate_probability(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1. Received: {value}")


def resolve_graph_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / "graph" / filename,
        BASE_DIR / "manifest_generation" / "graph" / filename,
        BASE_DIR / "Graph_and_manifest" / filename,
        BASE_DIR / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find '{filename}'. Searched:\n - {searched}")


def resolve_output_base(filename: str) -> Path:
    graph_dir = BASE_DIR / "graph"
    if graph_dir.exists():
        return BASE_DIR / filename

    graph_dir = BASE_DIR / "Graph_and_manifest"
    if graph_dir.exists():
        return graph_dir / filename
    return BASE_DIR / filename


def load_input_files(nodes_path: Path, edges_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    nodes_df = pd.read_excel(nodes_path)
    edges_df = pd.read_excel(edges_path)

    missing_node_cols = REQUIRED_NODE_COLUMNS - set(nodes_df.columns)
    missing_edge_cols = REQUIRED_EDGE_COLUMNS - set(edges_df.columns)

    if missing_node_cols:
        raise ValueError(f"Nodes file is missing required columns: {sorted(missing_node_cols)}")
    if missing_edge_cols:
        raise ValueError(f"Edges file is missing required columns: {sorted(missing_edge_cols)}")

    node_ids = set(nodes_df["id"].astype(str))
    dangling_from = sorted(set(edges_df["from"].astype(str)) - node_ids)
    dangling_to = sorted(set(edges_df["to"].astype(str)) - node_ids)

    if dangling_from or dangling_to:
        raise ValueError(
            "Edges file references node IDs not present in the nodes file. "
            f"Missing from-nodes: {dangling_from[:5]} Missing to-nodes: {dangling_to[:5]}"
        )

    return nodes_df, edges_df


def build_undirected_graph(edges_df: pd.DataFrame) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for source, target in edges_df[["from", "to"]].astype(str).itertuples(index=False, name=None):
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
    return graph


def multi_source_shortest_distances(
    graph: Dict[str, Set[str]],
    sources: Iterable[str],
) -> Dict[str, int]:
    queue: deque[str] = deque()
    distances: Dict[str, int] = {}

    for source in sources:
        if source in distances:
            continue
        distances[source] = 0
        queue.append(source)

    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1

        for neighbour in graph.get(current, set()):
            if neighbour in distances:
                continue
            distances[neighbour] = next_distance
            queue.append(neighbour)

    return distances


def format_pax_id(x_coord: int, y_coord: int) -> str:
    return f"{int(x_coord):03d}{int(y_coord):02d}"


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


def infer_main_aisle_y_values(nodes_df: pd.DataFrame) -> Tuple[int, int]:
    aisle_nodes = nodes_df.loc[nodes_df["type"].astype(str).str.lower().eq("aisle")].copy()
    if aisle_nodes.empty:
        raise ValueError("No aisle nodes found in the nodes file.")

    aisle_counts = aisle_nodes["y"].astype(int).value_counts().sort_values(ascending=False)
    if len(aisle_counts) < 2:
        raise ValueError("Could not infer two main aisle bands from the aisle nodes.")

    main_aisle_y_values = sorted(int(value) for value in aisle_counts.index[:2])
    return main_aisle_y_values[0], main_aisle_y_values[1]


def get_preferred_spawn(travel_class: str) -> str:
    return "F" if travel_class == "business" else "M"


def get_alternative_spawn(travel_class: str, row_number: int) -> str:
    if travel_class == "business":
        return "M"
    return "F" if row_number <= 18 else "R"


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
    raise ValueError(f"Unsupported row number for zone_std: {row_number}")


def get_zone_outsidein(row_number: int, seat_depth_category: str) -> int:
    if row_number in BUSINESS_ROWS:
        return 1
    if seat_depth_category == "maximum":
        return 2
    if seat_depth_category == "intermediate":
        return 3
    return 4


def get_zone_pyramid(row_number: int, seat_letter: str, seat_depth_category: str) -> int:
    if row_number in BUSINESS_ROWS:
        return 1

    is_rear = 24 <= row_number <= 38
    is_forward = 8 <= row_number <= 23

    if is_rear and seat_depth_category == "maximum":
        return 2

    if (is_forward and seat_depth_category == "maximum") or (
        is_rear and seat_depth_category == "intermediate"
    ):
        return 3

    if (is_rear and seat_letter in REAR_OUTER_AISLE_ECONOMY_SEATS) or (
        14 <= row_number <= 23 and seat_depth_category == "intermediate"
    ):
        return 4

    if is_forward or is_rear:
        return 5

    raise ValueError(
        f"Could not assign zone_pyramid for row {row_number}, seat letter {seat_letter}."
    )


def build_seat_map(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    nodes_df = nodes_df.copy()
    nodes_df["id"] = nodes_df["id"].astype(str)
    nodes_df["type"] = nodes_df["type"].astype(str).str.lower()
    nodes_df["x"] = nodes_df["x"].astype(int)
    nodes_df["y"] = nodes_df["y"].astype(int)

    graph = build_undirected_graph(edges_df)
    seat_nodes = nodes_df.loc[nodes_df["type"].eq("seat")].copy()

    row_x_values = sorted(seat_nodes["x"].unique())
    x_to_row_number = {x_value: index + 1 for index, x_value in enumerate(row_x_values)}
    if len(x_to_row_number) != 38:
        raise ValueError(
            f"Expected 38 seat rows from the revised layout, but found {len(x_to_row_number)}."
        )

    left_aisle_y, right_aisle_y = infer_main_aisle_y_values(nodes_df)
    left_aisle_sources = nodes_df.loc[nodes_df["type"].eq("aisle") & nodes_df["y"].eq(left_aisle_y), "id"]
    right_aisle_sources = nodes_df.loc[nodes_df["type"].eq("aisle") & nodes_df["y"].eq(right_aisle_y), "id"]
    all_aisle_sources = nodes_df.loc[nodes_df["type"].eq("aisle"), "id"]

    left_distances = multi_source_shortest_distances(graph, left_aisle_sources.astype(str))
    right_distances = multi_source_shortest_distances(graph, right_aisle_sources.astype(str))
    any_aisle_distances = multi_source_shortest_distances(graph, all_aisle_sources.astype(str))

    seat_nodes["row_number"] = seat_nodes["x"].map(x_to_row_number)
    seat_nodes["seat_letter"] = seat_nodes.apply(
        lambda row: get_seat_letter(int(row["row_number"]), int(row["y"])), axis=1
    )
    seat_nodes["seat"] = seat_nodes["row_number"].astype(str) + seat_nodes["seat_letter"]
    seat_nodes["class"] = seat_nodes["row_number"].map(get_class_from_row)

    seat_nodes["dist_left_aisle"] = seat_nodes["id"].map(left_distances)
    seat_nodes["dist_right_aisle"] = seat_nodes["id"].map(right_distances)
    seat_nodes["dist_to_aisle"] = seat_nodes["id"].map(any_aisle_distances)

    if seat_nodes[["dist_left_aisle", "dist_right_aisle", "dist_to_aisle"]].isna().any().any():
        unreachable = seat_nodes.loc[
            seat_nodes[["dist_left_aisle", "dist_right_aisle", "dist_to_aisle"]].isna().any(axis=1),
            "id",
        ].tolist()[:10]
        raise ValueError(f"Some seat nodes are not connected to the aisle network: {unreachable}")

    seat_nodes["dist_left_aisle"] = seat_nodes["dist_left_aisle"].astype(int)
    seat_nodes["dist_right_aisle"] = seat_nodes["dist_right_aisle"].astype(int)
    seat_nodes["dist_to_aisle"] = seat_nodes["dist_to_aisle"].astype(int)

    seat_nodes["assigned_aisle"] = np.where(
        seat_nodes["dist_left_aisle"] <= seat_nodes["dist_right_aisle"],
        "L",
        "R",
    )

    row_max_distance = seat_nodes.groupby("row_number")["dist_to_aisle"].transform("max")
    seat_nodes["seat_depth_category"] = np.select(
        [
            seat_nodes["class"].eq("business"),
            seat_nodes["dist_to_aisle"].eq(1),
            seat_nodes["dist_to_aisle"].eq(row_max_distance),
        ],
        [
            "business_direct",
            "aisle",
            "maximum",
        ],
        default="intermediate",
    )

    seat_nodes["preferred_spawn"] = seat_nodes["class"].map(get_preferred_spawn)
    seat_nodes["preferred_door"] = seat_nodes["preferred_spawn"]
    seat_nodes["alternative_spawn"] = seat_nodes.apply(
        lambda row: get_alternative_spawn(str(row["class"]), int(row["row_number"])),
        axis=1,
    )
    seat_nodes["zone_std"] = seat_nodes["row_number"].map(get_zone_std)
    seat_nodes["zone_outsidein"] = seat_nodes.apply(
        lambda row: get_zone_outsidein(int(row["row_number"]), str(row["seat_depth_category"])),
        axis=1,
    )
    seat_nodes["zone_pyramid"] = seat_nodes.apply(
        lambda row: get_zone_pyramid(
            int(row["row_number"]),
            str(row["seat_letter"]),
            str(row["seat_depth_category"]),
        ),
        axis=1,
    )
    seat_nodes["pax_id"] = seat_nodes.apply(
        lambda row: format_pax_id(int(row["x"]), int(row["y"])), axis=1
    )

    duplicate_checks = {
        "seat": seat_nodes["seat"],
        "pax_id": seat_nodes["pax_id"],
    }
    for label, series in duplicate_checks.items():
        if series.duplicated().any():
            duplicates = series.loc[series.duplicated()].tolist()[:10]
            raise ValueError(f"Duplicate {label} values detected: {duplicates}")

    return seat_nodes.sort_values(["row_number", "y"]).reset_index(drop=True)


def choose_occupied_seats(
    seat_map_df: pd.DataFrame,
    global_load_factor: float,
    business_load_factor: Optional[float],
    premium_load_factor: Optional[float],
    economy_load_factor: Optional[float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    class_specific = {
        "business": business_load_factor,
        "premium_economy": premium_load_factor,
        "economy": economy_load_factor,
    }

    occupied_parts: List[pd.DataFrame] = []

    for travel_class, class_df in seat_map_df.groupby("class", sort=False):
        load_factor = class_specific[travel_class]
        if load_factor is None:
            load_factor = global_load_factor

        n_class_seats = len(class_df)
        n_to_sample = int(round(n_class_seats * load_factor))
        n_to_sample = min(max(n_to_sample, 0), n_class_seats)

        if n_to_sample == 0:
            continue

        sampled_indices = rng.choice(class_df.index.to_numpy(), size=n_to_sample, replace=False)
        occupied_parts.append(class_df.loc[sampled_indices])

    if not occupied_parts:
        raise ValueError("No occupied seats were selected. Increase the load factor(s).")

    occupied_df = pd.concat(occupied_parts, ignore_index=True)
    return occupied_df.sort_values(["row_number", "seat_letter"]).reset_index(drop=True)


def sample_stow_seconds(
    rng: np.random.Generator,
    size: int,
    stow_config: StowConfig,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=float)

    dist = stow_config.dist.strip().lower()
    if dist == "uniform":
        low_s, high_s = validate_bounds(stow_config.low_s, stow_config.high_s, "stow uniform")
        samples = rng.uniform(low_s, high_s, size=size)
    elif dist == "weibull":
        if stow_config.shape_k <= 0 or stow_config.scale_lambda_s <= 0:
            raise ValueError(
                "Weibull stow parameters must be > 0: "
                f"shape={stow_config.shape_k}, scale={stow_config.scale_lambda_s}"
            )
        samples = rng.weibull(stow_config.shape_k, size=size) * stow_config.scale_lambda_s
    else:
        raise ValueError(f"Unsupported stow distribution: {stow_config.dist}")

    # Keep duration strictly positive to avoid zero-tick stowing.
    return np.maximum(samples, 1.0)


def build_manifest(
    seat_map_df: pd.DataFrame,
    global_load_factor: float,
    business_load_factor: Optional[float],
    premium_load_factor: Optional[float],
    economy_load_factor: Optional[float],
    rng: np.random.Generator,
    luggage_probability: float = LUGGAGE_PROBABILITY,
    stow_config: StowConfig = DEFAULT_STOW_CONFIG,
) -> pd.DataFrame:
    manifest_df = choose_occupied_seats(
        seat_map_df=seat_map_df,
        global_load_factor=global_load_factor,
        business_load_factor=business_load_factor,
        premium_load_factor=premium_load_factor,
        economy_load_factor=economy_load_factor,
        rng=rng,
    ).copy()

    manifest_df["preferred_speed"] = rng.choice([2, 4], size=len(manifest_df))
    manifest_df["lateral_speed"] = 1
    manifest_df["has_luggage"] = rng.random(size=len(manifest_df)) < luggage_probability

    stow_duration = np.full(len(manifest_df), np.nan)
    luggage_mask = manifest_df["has_luggage"].to_numpy()
    stow_duration[luggage_mask] = sample_stow_seconds(
        rng,
        size=int(luggage_mask.sum()),
        stow_config=stow_config,
    )
    manifest_df["stow_duration"] = stow_duration

    manifest_df = manifest_df.rename(columns={"x": "x_coord", "y": "y_coord"})
    manifest_df["stow_duration"] = manifest_df["stow_duration"].round().astype("Int64")
    manifest_df["has_luggage"] = manifest_df["has_luggage"].astype(bool)

    return manifest_df.sort_values(["x_coord", "y_coord"])[OUTPUT_COLUMNS].reset_index(drop=True)


def write_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return

    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        export_df = df.copy()
        text_columns = {
            "pax_id",
            "seat",
            "class",
            "preferred_door",
            "preferred_spawn",
            "alternative_spawn",
            "assigned_aisle",
        }
        for column_name in text_columns.intersection(export_df.columns):
            export_df[column_name] = export_df[column_name].astype(str)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="manifest")
            worksheet = writer.sheets["manifest"]

            for column_index, column_name in enumerate(df.columns, start=1):
                if column_name in text_columns:
                    for cell_group in worksheet.iter_cols(
                        min_col=column_index,
                        max_col=column_index,
                        min_row=2,
                        max_row=len(df) + 1,
                    ):
                        for cell in cell_group:
                            cell.number_format = "@"
        return

    raise ValueError(
        f"Unsupported output extension '{output_path.suffix}'. Use .csv or .xlsx."
    )


def resolve_output_path(base_output: Path, manifest_number: int) -> Path:
    stem = base_output.stem
    suffix = base_output.suffix
    numbered_name = f"{stem}_{manifest_number}{suffix}"
    return base_output.with_name(numbered_name)


def main() -> None:
    validate_probability("LOAD_FACTOR", LOAD_FACTOR)
    validate_probability("BUSINESS_LOAD_FACTOR", BUSINESS_LOAD_FACTOR)
    validate_probability("PREMIUM_LOAD_FACTOR", PREMIUM_LOAD_FACTOR)
    validate_probability("ECONOMY_LOAD_FACTOR", ECONOMY_LOAD_FACTOR)
    validate_probability("LUGGAGE_PROBABILITY", LUGGAGE_PROBABILITY)

    nodes_path = resolve_graph_file(NODES_FILE)
    edges_path = resolve_graph_file(EDGES_FILE)
    output_base = resolve_output_base(OUTPUT_FILE)

    nodes_df, edges_df = load_input_files(nodes_path, edges_path)
    seat_map_df = build_seat_map(nodes_df, edges_df)

    for manifest_number in range(1, N_MANIFESTS + 1):
        current_seed = None if SEED is None else SEED + manifest_number - 1
        rng = np.random.default_rng(current_seed)

        manifest_df = build_manifest(
            seat_map_df=seat_map_df,
            global_load_factor=LOAD_FACTOR,
            business_load_factor=BUSINESS_LOAD_FACTOR,
            premium_load_factor=PREMIUM_LOAD_FACTOR,
            economy_load_factor=ECONOMY_LOAD_FACTOR,
            rng=rng,
            luggage_probability=LUGGAGE_PROBABILITY,
            stow_config=DEFAULT_STOW_CONFIG,
        )

        current_output_path = resolve_output_path(output_base, manifest_number)
        write_dataframe(manifest_df, current_output_path)

        print(
            f"Generated {len(manifest_df)} passengers in '{current_output_path}'. "
            f"Seed={current_seed}, load_factor={LOAD_FACTOR}"
        )


if __name__ == "__main__":
    main()
