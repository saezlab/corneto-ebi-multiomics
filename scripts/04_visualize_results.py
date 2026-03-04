"""
Network visualization and interpretation.

This script visualizes the CORNETO network inference results and
compares them with the published network from Tüchler et al. (2025).

It rebuilds a CORNETO Graph from the exported TSV files (network_edges.tsv,
network_nodes.tsv) and uses CORNETO's built-in signaling preset for plotting.
This decouples visualization from the optimization run in script 03.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import corneto as cn
from corneto.graph import Graph


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

# %% 2. Rebuild CORNETO Graph and Data from exported results
#
# We reconstruct the Graph from the edge table and the Data object from
# the node table, so we can use CORNETO's signaling preset for plotting.

edge_tuples = list(zip(edges["source"], edges["sign"], edges["target"]))
G = Graph.from_tuples(edge_tuples)

# Rebuild the Data object from node roles and values
sample_data = {}
for _, row in nodes.iterrows():
    if row["type"] in ("input", "output"):
        sample_data[row["node"]] = {
            "value": float(row["value"]),
            "mapping": "vertex",
            "role": row["type"],
        }
data = cn.Data.from_cdict({"early": sample_data})

# Map solution values to the Graph's vertex/edge order
vertex_value_map = dict(zip(nodes["node"], nodes["value"]))
vertex_values = [vertex_value_map.get(name, 0.0) for name in G.V]
edge_values = list(edges["edge_value"])

# %% 3. Network visualization with CORNETO
#
# The signaling preset colors nodes and edges by their solution values
# (red = positive/activated, blue = negative/inhibited), and uses the
# PKN sign for arrow style (normal = activation, tee = inhibition).
# feature_data annotates nodes by role (input/output).

g = G.plot(
    preset="signaling",
    feature_data=data,
    solution={
        "v": vertex_values,
        "e": edge_values,
    },
    solution_map={"vertex": "v", "edge": "e"},
)

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

# %% 5. Plot the published network with CORNETO
#
# We use the same signaling preset as for our network, so the two plots
# are directly comparable. The paper's sign column is "(1)" or "(-1)",
# which we parse to integer. Node types are mapped to input/output roles:
# TF and Kinase/phosphatase → input, Secreted proteins → output, PKN → intermediate.

paper_early_sign = paper_early["sign"].str.strip("()").astype(int)
paper_edge_tuples = list(zip(paper_early["source"], paper_early_sign, paper_early["target"]))
G_paper = Graph.from_tuples(paper_edge_tuples)

# Map paper node types to CORNETO roles
PAPER_ROLE_MAP = {
    "TF": "input",
    "Kinase/ phosphatase": "input",
    "Secreted proteins": "output",
}

paper_sample_data = {}
for _, row in paper_early_nodes.iterrows():
    role = PAPER_ROLE_MAP.get(row["type"])
    if role:
        paper_sample_data[row["node"]] = {
            "value": float(row["value"]),
            "mapping": "vertex",
            "role": role,
        }
paper_data = cn.Data.from_cdict({"early": paper_sample_data})

paper_vertex_map = dict(zip(paper_early_nodes["node"], paper_early_nodes["value"]))
paper_vertex_values = [float(paper_vertex_map.get(name, 0.0)) for name in G_paper.V]
paper_edge_values = list(paper_early_sign)

g_paper = G_paper.plot(
    preset="signaling",
    feature_data=paper_data,
    solution={
        "v": paper_vertex_values,
        "e": paper_edge_values,
    },
    solution_map={"vertex": "v", "edge": "e"},
)

g_paper.render(RESULTS_DIR / "network_paper_early", format="pdf", cleanup=True)
g_paper.render(RESULTS_DIR / "network_paper_early", format="png", cleanup=True)
print(f"\nSaved paper network plot to {RESULTS_DIR / 'network_paper_early.pdf'}")

# %% 6. Node degree distribution

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
