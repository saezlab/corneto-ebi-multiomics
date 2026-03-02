"""
Network visualization and interpretation.

This script visualizes the CORNETO network inference results and
compares them with the published network from Tüchler et al. (2025).
"""

import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# %% Setting up paths for data and results
# this is not important, it's we only needed to allow us to run the same code
# either as a script or in an interactive session

try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results" if (_script_root / "data").is_dir() else Path("results")

# %% 1. Load results

edges = pd.read_csv(RESULTS_DIR / "network_edges.tsv", sep="\t")
nodes = pd.read_csv(RESULTS_DIR / "network_nodes.tsv", sep="\t")

print(f"Network: {len(edges)} edges, {len(nodes)} nodes")

# %% 2. Load input data for annotation

activities_early = pd.read_csv(DATA_DIR / "differential" / "activities_early.tsv", sep="\t")
secretome_early = pd.read_csv(DATA_DIR / "differential" / "secretome_early.tsv", sep="\t")

input_nodes = set(activities_early["source"]) | {"TGFB1"}
output_nodes = set(secretome_early["id"])

# %% 3. Network visualization with graphviz
#
# Color scheme:
# - Red fill: perturbation / stimulus (TGFB1)
# - Pink fill: perturbation nodes (TF/kinase activities)
# - Light green fill: measurement nodes (secretome)
# - White: intermediate signaling nodes
#
# Edge colors:
# - Red: activation signal
# - Blue: inhibition signal
#
# Arrow styles:
# - Normal arrow: activation (+1)
# - Tee (flat end): inhibition (-1)


def build_network_graph(edges_df, nodes_df, input_nodes, output_nodes):
    """
    Build a graphviz Digraph from edge and node tables.
    """

    g = graphviz.Digraph(
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fillcolor": "white",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
        },
    )

    # Determine all nodes in the network
    network_nodes = set(edges_df["source"]) | set(edges_df["target"])

    # Add nodes with type-specific colors
    for node_name in network_nodes:
        if node_name == "TGFB1":
            g.node(node_name, fillcolor="#ff6b6b", fontcolor="white")
        elif node_name in input_nodes:
            g.node(node_name, fillcolor="#ffb3b3")
        elif node_name in output_nodes:
            g.node(node_name, fillcolor="#b3e6b3")
        else:
            g.node(node_name, fillcolor="#f0f0f0")

    # Add edges with sign-specific styling
    for _, row in edges_df.iterrows():
        edge_color = "red" if row["edge_value"] > 0 else "blue"
        arrowhead = "normal" if row["sign"] > 0 else "tee"
        g.edge(row["source"], row["target"],
               color=edge_color, arrowhead=arrowhead)

    return g


g = build_network_graph(edges, nodes, input_nodes, output_nodes)

# Render to file
g.render(RESULTS_DIR / "network", format="pdf", cleanup=True)
g.render(RESULTS_DIR / "network", format="png", cleanup=True)
print(f"Saved network visualization to {RESULTS_DIR / 'network.pdf'}")

# Display (if in interactive environment)
g

# %% 4. Compare with published network
#
# Load the published network edges from the paper's supplementary data.
# Note: the paper's "early" network is the union of phases 1 (initial:
# TGFB1 → activities) and 2 (early: activities → secretome). Our tutorial
# runs only phase 2, so the comparison is approximate.

paper_edges = pd.read_csv(DATA_DIR / "network" / "paper_edges.tsv", sep="\t")
paper_nodes = pd.read_csv(DATA_DIR / "network" / "paper_nodes.tsv", sep="\t")

# Filter for the early network
paper_early = paper_edges[paper_edges["network"] == "early"]
paper_early_nodes = paper_nodes[paper_nodes["network"] == "early"]

print(f"\nPublished early network: {len(paper_early)} edges, "
      f"{len(paper_early_nodes)} nodes")

# Parse published edges: the 'sign' column contains e.g. "(1)" or "(-1)"
paper_early_pairs = set(zip(paper_early["source"], paper_early["target"]))
our_pairs = set(zip(edges["source"], edges["target"]))

overlap = paper_early_pairs & our_pairs
only_paper = paper_early_pairs - our_pairs
only_ours = our_pairs - paper_early_pairs

print(f"\nEdge comparison (our network vs published early network):")
print(f"  Shared edges: {len(overlap)}")
print(f"  Only in published: {len(only_paper)}")
print(f"  Only in ours: {len(only_ours)}")

if len(paper_early_pairs) > 0:
    jaccard = len(overlap) / len(paper_early_pairs | our_pairs)
    print(f"  Jaccard similarity: {jaccard:.3f}")

# Node overlap
paper_early_node_set = set(paper_early_nodes["node"])
our_node_set = set(nodes["node"])
node_overlap = paper_early_node_set & our_node_set

print(f"\nNode comparison:")
print(f"  Shared nodes: {len(node_overlap)}")
print(f"  Only in published: {len(paper_early_node_set - our_node_set)}")
print(f"  Only in ours: {len(our_node_set - paper_early_node_set)}")

# %% 5. Node degree distribution

if len(edges) > 0:
    degree = pd.concat([
        edges["source"].value_counts().rename("out_degree"),
        edges["target"].value_counts().rename("in_degree"),
    ], axis=1).fillna(0).astype(int)
    degree["total_degree"] = degree["out_degree"] + degree["in_degree"]
    degree = degree.sort_values("total_degree", ascending=False)

    print(f"\nTop 10 hub nodes:")
    print(degree.head(10))

    # Plot degree distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(degree["total_degree"], bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Count")
    ax.set_title("Node degree distribution in inferred network")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_distribution.pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% 6. Optional: overlay with collagen imaging data
#
# The collagen I imaging data shows the phenotypic outcome (ECM deposition)
# over time. We can check whether key network nodes are associated with
# the timing of collagen accumulation.

imaging = pd.read_csv(DATA_DIR / "imaging" / "col1_timecourse.tsv", sep="\t")

# At 0h there is no TGF-beta data (only control). Duplicate control as TGF
# baseline so both conditions start from the same point.
ctrl_0h = imaging[imaging["time"] == "0h"].copy()
ctrl_0h["condition"] = "TGF"
imaging = pd.concat([imaging, ctrl_0h], ignore_index=True)

# Summarize COL1 intensity by time and condition
time_order = ["0h", "12h", "24h", "48h", "72h", "96h"]

col1_summary = (
    imaging
    .groupby(["time", "condition"])
    ["Col1_per_cell"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7, 4))
for condition, grp in col1_summary.groupby("condition"):
    grp = grp.set_index("time").loc[time_order].reset_index()
    ax.errorbar(
        grp["time"], grp["mean"],
        yerr=grp["std"] / np.sqrt(grp["count"]),
        marker="o", label=condition, capsize=3,
    )
ax.set_xlabel("Time after treatment")
ax.set_ylabel("COL1 per cell (mean intensity)")
ax.set_title("Collagen I deposition over time")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "col1_timecourse.pdf", bbox_inches="tight")
plt.savefig(RESULTS_DIR / "col1_timecourse.png", dpi=150, bbox_inches="tight")
print(f"\nSaved COL1 plot to {RESULTS_DIR / 'col1_timecourse.pdf'}")
plt.show()

# %% Notes
#
# Key things to discuss in the tutorial:
#
# 1. How does the network topology change with different lambda_reg values?
# 2. Which nodes are consistent between our result and the published network?
# 3. What biological pathways are represented in the inferred network?
# 4. How do the early and late networks differ (if running both)?
# 5. What are the limitations of this approach?
#    - Depends on PKN completeness
#    - Single optimal solution (not the full solution space)
#    - Static snapshot of a dynamic process
