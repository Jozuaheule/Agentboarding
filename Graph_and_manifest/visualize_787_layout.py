from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math


# Optional toggle for drawing node labels
SHOW_NODE_LABELS = False
X_AXIS_TICK_STEP = 2


NODE_TYPE_COLORS = {
    "aisle": "lightgray",
    "seat": "blue",
    "galley": "gray",
}


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input_path(filename):
    return SCRIPT_DIR / filename


def load_graph(nodes_file="nodes_787.xlsx", edges_file="edges_787.xlsx"):
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
            x=row[x_col],
            y=row[y_col],
            type=node_type,
        )

    for _, row in df_edges.iterrows():
        u = row["from"]
        v = row["to"]
        length = row["length"]
        graph.add_edge(u, v, length=length)

    return graph


def plot_graph(graph):
    pos = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}

    plt.figure(figsize=(14, 6))

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="darkgray",
        width=0.6,
        arrows=False,
    )

    for node_type, color in NODE_TYPE_COLORS.items():
        nodelist = [n for n, d in graph.nodes(data=True) if d.get("type") == node_type]
        if nodelist:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodelist,
                node_color=color,
                node_size=28,
                label=node_type,
            )

    if SHOW_NODE_LABELS:
        labels = {n: str(n) for n in graph.nodes}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

    plt.axis("equal")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(X_AXIS_TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_min, x_max = math.floor(min(x_vals)), math.ceil(max(x_vals))
    y_min, y_max = math.floor(min(y_vals)), math.ceil(max(y_vals))
    ax.set_xticks(range(x_min, x_max + 1, X_AXIS_TICK_STEP))
    ax.set_yticks(range(y_min, y_max + 1, 1))
    ax.tick_params(axis="both", which="major", labelsize=9, labelbottom=True, labelleft=True)
    plt.xlabel("x (longitudinal)")
    plt.ylabel("y (lateral)")
    plt.title("B787-inspired Custom Cabin Layout - Network Graph")
    plt.legend(title="Node Type")
    plt.tight_layout()
    plt.show()


def main():
    graph = load_graph("nodes_787.xlsx", "edges_787.xlsx")
    plot_graph(graph)


if __name__ == "__main__":
    main()
