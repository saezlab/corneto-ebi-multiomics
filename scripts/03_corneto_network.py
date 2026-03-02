"""
CARNIVAL network inference with CORNETO.

This script uses the CARNIVAL algorithm (implemented in CORNETO) to find
an optimal subnetwork of the prior knowledge network that connects
perturbations to measurements.

In our "early" model (corresponding to model 2 in the paper):
- Perturbations (inputs): TGFB1 stimulus + early TF/kinase activities
- Measurements (outputs): early secretome fold changes

CARNIVAL solves an integer linear program that:
- Maximizes consistency between network edges and observed data
- Minimizes network size (L0 penalty on active edges)
"""

import pandas as pd
import numpy as np
from pathlib import Path

import corneto as cn
from corneto.methods.future.carnival import CarnivalFlow
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
RESULTS_DIR.mkdir(exist_ok=True)

# %% 1. Load prepared inputs

pkn = pd.read_csv(DATA_DIR / "network" / "pkn.tsv", sep="\t")
activities_early = pd.read_csv(DATA_DIR / "differential" / "activities_early.tsv", sep="\t")
secretome_early = pd.read_csv(DATA_DIR / "differential" / "secretome_early.tsv", sep="\t")

print(f"PKN: {len(pkn)} interactions, "
      f"{len(set(pkn['source']) | set(pkn['target']))} nodes")
print(f"Perturbation nodes: {len(activities_early)}")
print(f"Measurements (secretome): {len(secretome_early)}")

# %% 2. Build CORNETO Graph from PKN
#
# The Graph object represents the prior knowledge network.
# Each edge is a tuple: (source, sign, target)

edge_tuples = list(zip(pkn["source"], pkn["mor"], pkn["target"]))
G = cn.graph.Graph.from_tuples(edge_tuples)

print(f"\nCORNETO Graph: {G.num_vertices} vertices, {G.num_edges} edges")

# %% 3. Prepare perturbation and measurement data
#
# CORNETO expects a dictionary of samples, where each sample maps
# node names to their role (input/output) and observed value.
#
# - Inputs (perturbations): TGFB1 stimulus and TF/kinase activity scores
# - Outputs (measurements): secretome fold changes to be explained

sample_data = {}

# Add TGFB1 as the known stimulus
sample_data["TGFB1"] = {"value": 1.0, "mapping": "vertex", "role": "input"}

# Add enzyme activities as perturbations
for _, row in activities_early.iterrows():
    sample_data[row["source"]] = {
        "value": float(row["score"]),
        "mapping": "vertex",
        "role": "input",
    }

# Add secretome as measurements
for _, row in secretome_early.iterrows():
    sample_data[row["id"]] = {
        "value": float(row["score"]),
        "mapping": "vertex",
        "role": "output",
    }

data = cn.Data.from_cdict({"early": sample_data})

n_inputs = sum(1 for v in sample_data.values() if v["role"] == "input")
n_outputs = sum(1 for v in sample_data.values() if v["role"] == "output")
print(f"\nData: {n_inputs} inputs, {n_outputs} outputs")

# %% 4. Build CARNIVAL optimization problem
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
# CarnivalFlow is a specific CARNIVAL formulation that models signal
# propagation as network flow, which makes the optimization more
# efficient for large networks.

LAMBDA_REG = 0.01

carnival = CarnivalFlow(lambda_reg=LAMBDA_REG)
problem = carnival.build(G, data)

print(f"\nCARNIVAL problem built (lambda_reg={LAMBDA_REG})")

# %% 5. Solve
#
# The problem is a Mixed Integer Program (MIP): some variables are
# continuous (flow values), others are binary (is this edge active?).
# MIP problems are NP-hard in general, but modern solvers handle
# networks of this size well.
#
# We use SCIP, a powerful open-source MIP solver. CVXPY translates
# our problem into SCIP's format and calls it. Other options:
# - "highs": another open-source solver, sometimes faster
# - "scipy": simpler, works for small problems
#
# The solver output is verbose but informative. Here's a quick guide
# to what you'll see:
#
# 1. CVXPY compilation: CVXPY converts the problem into standard form.
#    "Your problem has N variables, M constraints" tells you the size.
#
# 2. Presolving: SCIP simplifies the problem before solving — removing
#    redundant variables and constraints, tightening bounds. Lines like
#    "959 del vars, 158226 del conss" mean it eliminated that many.
#    This can take 30-60s but dramatically speeds up the actual solve.
#
# 3. Solving: The main table shows progress. Key columns:
#    - time: elapsed seconds
#    - node/left: search tree progress (branch-and-bound exploration)
#    - dualbound: best possible objective (lower bound)
#    - primalbound: best feasible solution found so far (upper bound)
#    - gap: relative difference between the two bounds; 0% = optimal
#
#    Lines starting with "o" or "d" mark new best solutions found.
#    The solver is done when gap reaches 0% or the time limit is hit.

SOLVER = "SCIP"
TIME_LIMIT = 300  # seconds

print(f"\nSolving with {SOLVER} (time limit: {TIME_LIMIT}s)...")
problem.solve(solver=SOLVER, verbosity=1, **{"limits/time": TIME_LIMIT})

# Objective 0: data fit (sign mismatches); Objective 1: network size (edge count)
print("\nObjective values:")
for i, obj in enumerate(problem.objectives):
    print(f"  Objective {i}: {obj.value}")

# %% 6. Visualize with CORNETO's built-in plotting
#
# CORNETO can plot the solved network directly using its plotting utilities.
# cn.pl.edge_style and cn.pl.vertex_style extract graphviz attributes
# from the solved problem, coloring nodes and edges by their sign
# (red = activated, blue = inhibited). edge_indexes filters to active edges.

active_edges = list(np.flatnonzero(problem.expr.edge_has_signal.value))

g = carnival.processed_graph.plot(
    custom_edge_attr=cn.pl.edge_style(problem, edge_var="edge_value"),
    custom_vertex_attr=cn.pl.vertex_style(problem, carnival.processed_graph,
                                          vertex_var="vertex_value"),
    edge_indexes=active_edges,
)

# g is a graphviz.Digraph: in a notebook it renders inline,
# from the terminal we save it to file
g.render(RESULTS_DIR / "network_corneto", format="pdf", cleanup=True)
print(f"Saved network plot to {RESULTS_DIR / 'network_corneto.pdf'}")

# %% 7. Extract results
#
# Get edge and vertex activity values from the solution.

edge_values = problem.expr.edge_value.value.flatten()
vertex_values = problem.expr.vertex_value.value.flatten()

# Build edge result table
# CARNIVAL adds auxiliary edges for perturbations/measurements beyond
# the original PKN edges; we only extract the original PKN edges.
edges_result = []

for i, (src, sign, tgt) in enumerate(edge_tuples):
    if abs(edge_values[i]) > 1e-6:  # active edges only
        edges_result.append({
            "source": src,
            "sign": int(sign),
            "target": tgt,
            "edge_value": float(edge_values[i]),
        })


edges_df = pd.DataFrame(edges_result).sort_values(["source", "target"]).reset_index(drop=True)
print(f"\nActive edges: {len(edges_df)}")

# Build node result table
vertex_names = G.V  # vertex names (tuple)
nodes_result = []

for i, name in enumerate(vertex_names):
    if abs(vertex_values[i]) > 1e-6:  # active nodes only
        node_type = "intermediate"
        if name in sample_data:
            node_type = sample_data[name]["role"]
        nodes_result.append({
            "node": name,
            "value": float(vertex_values[i]),
            "type": node_type,
        })

nodes_df = pd.DataFrame(nodes_result).sort_values("node").reset_index(drop=True)
print(f"Active nodes: {len(nodes_df)}")

# %% 8. Save results

edges_df.to_csv(RESULTS_DIR / "network_edges.tsv", sep="\t", index=False)
nodes_df.to_csv(RESULTS_DIR / "network_nodes.tsv", sep="\t", index=False)

print(f"\nSaved results to {RESULTS_DIR}:")
print(f"  network_edges.tsv ({len(edges_df)} edges)")
print(f"  network_nodes.tsv ({len(nodes_df)} nodes)")

# %% 9. Quick summary statistics

if len(edges_df) > 0:
    print(f"\nNetwork summary:")
    print(f"  Activating edges: {(edges_df['sign'] > 0).sum()}")
    print(f"  Inhibiting edges: {(edges_df['sign'] < 0).sum()}")
    print(f"  Input nodes: {(nodes_df['type'] == 'input').sum()}")
    print(f"  Output nodes: {(nodes_df['type'] == 'output').sum()}")
    print(f"  Intermediate nodes: {(nodes_df['type'] == 'intermediate').sum()}")
else:
    print("\nNo active edges found. Consider adjusting lambda_reg or solver settings.")

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
# The next script (04) will visualize the resulting network.
