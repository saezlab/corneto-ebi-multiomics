"""
CARNIVAL network inference with CORNETO.

This script uses the CARNIVAL algorithm (implemented in CORNETO) to find
optimal subnetworks of the prior knowledge network that connect
perturbations to measurements.

We run two models following the paper:
- Model 1 (TGFB1 → early activities): how does TGFB1 drive TF/kinase changes?
- Model 2 (activities → early secretome): how do TF/kinase activities explain secretome?

The two results are merged into a combined early network for comparison
with the published network.

CARNIVAL solves an integer linear program that:
- Maximizes consistency between network edges and observed data
- Minimizes network size (L0 penalty on active edges)
"""

import pandas as pd
from pathlib import Path

from carnival_utils import (
    run_carnival, extract_results, plot_network,
    save_results, merge_networks, print_summary,
)

# %% Setting up paths for data and results
# this is not important, it's we only needed to allow us to run the same code
# either as a script or in an interactive session

try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results" if (_script_root / "data").is_dir() else Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# %% 1. Parameters
#
# CARNIVAL finds the smallest subnetwork of the PKN that is logically
# consistent with the observed perturbation and measurement data.
# The optimization has two objectives:
#
#   Objective 0 (data fit): how well do the network edge signs agree
#       with the observed activity scores? Each measurement node has
#       an observed sign (+/-); the network should propagate signals
#       from perturbations to measurements without contradictions.
#       Lower = better fit (fewer sign mismatches).
#
#   Objective 1 (network size): how many edges are active? This is the
#       L0 regularization term — "L0" simply means we count the number
#       of non-zero (active) edges. We want a parsimonious network:
#       the simplest explanation consistent with the data.
#
# lambda_reg controls the trade-off between these two objectives:
# - Higher lambda_reg -> sparser network (fewer edges, possibly worse fit)
# - Lower lambda_reg -> denser network (better fit, harder to interpret)
#
# Typical values range from 1e-4 to 1. Values around 0.01-0.1 are most common.
# Core biology (e.g. TGFB1-SMAD cascade) is usually stable across this range,
# while peripheral edges come and go. It is worth trying a few values (e.g.
# 0.001, 0.01, 0.1; if measurements values are between -1 and 1, but larger
# values of lambda can be used) to see how the network changes. The choice of
# input data (significance thresholds, which omics to include) often has a
# larger impact than lambda_reg tuning.
#
# CarnivalFlow is a specific CARNIVAL formulation that models signal
# propagation as network flow, which makes the optimization more
# efficient for large networks.

LAMBDA_REG = 0.001
SOLVER = "SCIP"
TIME_LIMIT = 300  # seconds

# %% 2. Load prepared inputs

activities_early = pd.read_csv(DATA_DIR / "differential" / "activities_early.tsv", sep="\t")
secretome_early = pd.read_csv(DATA_DIR / "differential" / "secretome_early.tsv", sep="\t")

# %% 3. Model 1: TGFB1 → early activities
#
# This model asks: how does the TGFB1 stimulus propagate through the
# signaling network to explain the observed TF/kinase activity changes?
# TGFB1 is the only input; enzyme activities are the outputs.

print("=" * 60)
print("MODEL 1: TGFB1 → early activities")
print("=" * 60)

pkn_m1 = pd.read_csv(DATA_DIR / "network" / "pkn_model1.tsv", sep="\t")

sample_data_m1 = {
    "TGFB1": {"value": 1.0, "mapping": "vertex", "role": "input"},
}
for _, row in activities_early.iterrows():
    if row["source"] in set(pkn_m1["source"]) | set(pkn_m1["target"]):
        sample_data_m1[row["source"]] = {
            "value": float(row["score"]),
            "mapping": "vertex",
            "role": "output",
        }

carnival_m1, problem_m1, data_m1, _ = run_carnival(
    pkn_m1, sample_data_m1, lambda_reg=LAMBDA_REG,
    solver=SOLVER, time_limit=TIME_LIMIT,
)
edges_m1, nodes_m1 = extract_results(carnival_m1, problem_m1, sample_data_m1)

print("\nModel 1 result:")
print_summary(edges_m1, nodes_m1)

save_results(edges_m1, nodes_m1, "model1", RESULTS_DIR)

g_m1 = plot_network(edges_m1, nodes_m1)
g_m1.render(RESULTS_DIR / "network_model1", format="pdf", cleanup=True)
print(f"Saved model 1 plot to {RESULTS_DIR / 'network_model1.pdf'}")

# %% 4. Model 2: activities + TGFB1 → early secretome
#
# This model asks: given the observed TF/kinase activities (plus TGFB1),
# which signaling paths explain the secretome changes? Activities and
# TGFB1 are inputs; secretome fold changes are the outputs.

print("\n" + "=" * 60)
print("MODEL 2: activities + TGFB1 → early secretome")
print("=" * 60)

pkn_m2 = pd.read_csv(DATA_DIR / "network" / "pkn_model2.tsv", sep="\t")
pkn_m2_nodes = set(pkn_m2["source"]) | set(pkn_m2["target"])

sample_data_m2 = {
    "TGFB1": {"value": 1.0, "mapping": "vertex", "role": "input"},
}
for _, row in activities_early.iterrows():
    if row["source"] in pkn_m2_nodes:
        sample_data_m2[row["source"]] = {
            "value": float(row["score"]),
            "mapping": "vertex",
            "role": "input",
        }
for _, row in secretome_early.iterrows():
    if row["id"] in pkn_m2_nodes:
        sample_data_m2[row["id"]] = {
            "value": float(row["score"]),
            "mapping": "vertex",
            "role": "output",
        }

carnival_m2, problem_m2, data_m2, _ = run_carnival(
    pkn_m2, sample_data_m2, lambda_reg=LAMBDA_REG,
    solver=SOLVER, time_limit=TIME_LIMIT,
)
edges_m2, nodes_m2 = extract_results(carnival_m2, problem_m2, sample_data_m2)

print("\nModel 2 result:")
print_summary(edges_m2, nodes_m2)

save_results(edges_m2, nodes_m2, "model2", RESULTS_DIR)

g_m2 = plot_network(edges_m2, nodes_m2)
g_m2.render(RESULTS_DIR / "network_model2", format="pdf", cleanup=True)
print(f"Saved model 2 plot to {RESULTS_DIR / 'network_model2.pdf'}")

# %% 5. Merge models into combined early network
#
# The paper's "early" network is the union of models 1 and 2. We merge
# the two result networks: edges are deduplicated by (source, sign, target),
# nodes keep the value with the largest absolute magnitude.

print("\n" + "=" * 60)
print("MERGED: combined early network")
print("=" * 60)

edges_merged, nodes_merged = merge_networks(
    [edges_m1, edges_m2],
    [nodes_m1, nodes_m2],
)

print("\nMerged network:")
print_summary(edges_merged, nodes_merged)

save_results(edges_merged, nodes_merged, "network", RESULTS_DIR)

g_merged = plot_network(edges_merged, nodes_merged)
g_merged.render(RESULTS_DIR / "network_merged", format="pdf", cleanup=True)
print(f"Saved merged network plot to {RESULTS_DIR / 'network_merged.pdf'}")

# %% 6. How to read the network plot
#
# The signaling preset uses colors and arrow styles to encode the solution:
#
#   Node color: red = +1 (active/upregulated), blue = -1 (inhibited)
#   Edge color: red = +1, blue = -1 (represents: source_value × interaction_sign)
#   Arrow style: normal = activation (PKN sign +1), tee/hammer = inhibition (-1)
#
# In a logically consistent solution, the edge color always matches the
# TARGET node color. The four consistent patterns:
#
#   Source  Arrow (PKN)  Edge color  Target  Meaning
#   red     normal       red         red     Active node activates target
#   red     tee          blue        blue    Active node inhibits target
#   blue    normal       blue        blue    Inactive source propagates inactivity
#   blue    tee          red         red     Double negative: inactive + inhibition = active
#
# If the edge color differs from the target node color, that's a sign
# inconsistency. CARNIVAL allows some of these when a perfectly consistent
# solution is not possible — Objective 0 counts these mismatches.

# %% Notes
#
# If the network is too sparse, try:
#   - Decreasing lambda_reg (e.g., 0.001)
#   - Increasing the time limit
#   - Using a different solver (e.g., "scip" for better MIP performance)
#
# If the network is too dense, try:
#   - Increasing lambda_reg (e.g., 0.1)
#   - Applying stricter significance thresholds for inputs/outputs
#
# The next script (04) will visualize the resulting network and compare
# with the published network from the paper.
