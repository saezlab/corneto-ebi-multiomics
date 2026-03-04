"""
Shared helper functions for CARNIVAL network inference with CORNETO.

These functions wrap the common steps of building, solving, extracting,
saving/loading, plotting, and merging CARNIVAL network results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import corneto as cn
from corneto.methods.future.carnival import CarnivalFlow
from corneto.graph import Graph


def run_carnival(pkn_df, sample_data, lambda_reg=0.01, solver="SCIP",
                 time_limit=300):
    """Build CORNETO graph and data, solve CARNIVAL.

    Args:
        pkn_df: DataFrame with columns (source, mor, target).
        sample_data: dict mapping node names to
            {"value": float, "mapping": "vertex", "role": "input"|"output"}.
        lambda_reg: sparsity penalty (typical range 1e-4 to 1).
        solver: ILP solver name ("SCIP", "HiGHS", etc.).
        time_limit: solver time limit in seconds.

    Returns:
        Tuple of (carnival, problem, data, edge_tuples).
    """
    edge_tuples = list(zip(pkn_df["source"], pkn_df["mor"], pkn_df["target"]))
    G = Graph.from_tuples(edge_tuples)

    data = cn.Data.from_cdict({"sample": sample_data})

    n_inputs = sum(1 for v in sample_data.values() if v["role"] == "input")
    n_outputs = sum(1 for v in sample_data.values() if v["role"] == "output")
    print(f"CORNETO Graph: {G.num_vertices} vertices, {G.num_edges} edges")
    print(f"Data: {n_inputs} inputs, {n_outputs} outputs")

    carnival = CarnivalFlow(lambda_reg=lambda_reg)
    problem = carnival.build(G, data)
    print(f"CARNIVAL problem built (lambda_reg={lambda_reg})")

    print(f"Solving with {solver} (time limit: {time_limit}s)...")
    solver_kwargs = {}
    if solver.upper() == "SCIP":
        solver_kwargs["limits/time"] = time_limit
    problem.solve(solver=solver, verbosity=1, **solver_kwargs)

    print("\nObjective values:")
    for i, obj in enumerate(problem.objectives):
        print(f"  Objective {i}: {obj.value}")

    return carnival, problem, data, edge_tuples


def extract_results(carnival, problem, sample_data):
    """Extract edge and node tables from a solved CARNIVAL problem.

    Iterates over carnival.processed_graph (not the original PKN) to
    correctly handle auxiliary edges added by CARNIVAL.

    Returns:
        Tuple of (edges_df, nodes_df).
    """
    pg = carnival.processed_graph
    edge_values = problem.expr.edge_value.value.flatten()
    vertex_values = problem.expr.vertex_value.value.flatten()
    active_mask = problem.expr.edge_has_signal.value.flatten()

    edges_result = []
    for i in range(pg.num_edges):
        if abs(active_mask[i]) < 1e-6:
            continue
        src_set, tgt_set = pg.get_edge(i)
        if len(src_set) == 0 or len(tgt_set) == 0:
            continue
        src = list(src_set)[0]
        tgt = list(tgt_set)[0]
        sign = pg.get_attr_edge(i).get("interaction", 0)
        edges_result.append({
            "source": src,
            "sign": int(sign),
            "target": tgt,
            "edge_value": float(edge_values[i]),
        })

    edges_df = (pd.DataFrame(edges_result)
                .sort_values(["source", "target"])
                .reset_index(drop=True))

    nodes_result = []
    for i, name in enumerate(pg.V):
        if abs(vertex_values[i]) > 1e-6:
            node_type = "intermediate"
            if name in sample_data:
                node_type = sample_data[name]["role"]
            nodes_result.append({
                "node": name,
                "value": float(vertex_values[i]),
                "type": node_type,
            })

    nodes_df = (pd.DataFrame(nodes_result)
                .sort_values("node")
                .reset_index(drop=True))

    return edges_df, nodes_df


def plot_network(edges_df, nodes_df):
    """Rebuild a CORNETO Graph from result tables and plot with signaling preset.

    Returns:
        graphviz.Digraph object.
    """
    edge_tuples = list(zip(edges_df["source"], edges_df["sign"],
                           edges_df["target"]))
    G = Graph.from_tuples(edge_tuples)

    sample_data = {}
    for _, row in nodes_df.iterrows():
        if row["type"] in ("input", "output"):
            sample_data[row["node"]] = {
                "value": float(row["value"]),
                "mapping": "vertex",
                "role": row["type"],
            }
    data = cn.Data.from_cdict({"sample": sample_data})

    vertex_value_map = dict(zip(nodes_df["node"], nodes_df["value"]))
    vertex_values = [vertex_value_map.get(name, 0.0) for name in G.V]
    edge_values = list(edges_df["edge_value"])

    g = G.plot(
        preset="signaling",
        feature_data=data,
        solution={"v": vertex_values, "e": edge_values},
        solution_map={"vertex": "v", "edge": "e"},
    )
    return g


def save_results(edges_df, nodes_df, prefix, results_dir):
    """Save edge and node tables to TSV files."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    edges_df.to_csv(results_dir / f"{prefix}_edges.tsv", sep="\t", index=False)
    nodes_df.to_csv(results_dir / f"{prefix}_nodes.tsv", sep="\t", index=False)
    print(f"Saved {prefix}: {len(edges_df)} edges, {len(nodes_df)} nodes")


def load_results(prefix, results_dir):
    """Load edge and node tables from TSV files.

    Returns:
        Tuple of (edges_df, nodes_df).
    """
    results_dir = Path(results_dir)
    edges_df = pd.read_csv(results_dir / f"{prefix}_edges.tsv", sep="\t")
    nodes_df = pd.read_csv(results_dir / f"{prefix}_nodes.tsv", sep="\t")
    return edges_df, nodes_df


def merge_networks(edge_dfs, node_dfs):
    """Merge multiple network results.

    Edges are deduplicated by (source, sign, target); if the same edge
    appears in multiple networks, the edge_value from the first occurrence
    is kept. Nodes are deduplicated by name; if a node appears in multiple
    networks, the value with the largest absolute magnitude is kept, and
    the type priority is input > output > intermediate.

    Returns:
        Tuple of (merged_edges_df, merged_nodes_df).
    """
    all_edges = pd.concat(edge_dfs, ignore_index=True)
    merged_edges = (all_edges
                    .drop_duplicates(subset=["source", "sign", "target"])
                    .sort_values(["source", "target"])
                    .reset_index(drop=True))

    all_nodes = pd.concat(node_dfs, ignore_index=True)
    # For each node, keep the row with the largest |value|
    all_nodes["abs_value"] = all_nodes["value"].abs()
    merged_nodes = (all_nodes
                    .sort_values("abs_value", ascending=False)
                    .drop_duplicates(subset="node", keep="first")
                    .drop(columns="abs_value")
                    .sort_values("node")
                    .reset_index(drop=True))

    # Upgrade type: if a node is input in any network, keep it as input
    type_priority = {"input": 0, "output": 1, "intermediate": 2}
    best_type = {}
    for _, row in all_nodes.iterrows():
        name = row["node"]
        prio = type_priority.get(row["type"], 2)
        if name not in best_type or prio < best_type[name]:
            best_type[name] = prio
    inv_priority = {v: k for k, v in type_priority.items()}
    merged_nodes["type"] = merged_nodes["node"].map(
        lambda n: inv_priority[best_type[n]]
    )

    return merged_edges, merged_nodes


def print_summary(edges_df, nodes_df):
    """Print network summary statistics."""
    print(f"  Edges: {len(edges_df)} "
          f"({(edges_df['sign'] > 0).sum()} activating, "
          f"{(edges_df['sign'] < 0).sum()} inhibiting)")
    print(f"  Nodes: {len(nodes_df)} "
          f"({(nodes_df['type'] == 'input').sum()} input, "
          f"{(nodes_df['type'] == 'output').sum()} output, "
          f"{(nodes_df['type'] == 'intermediate').sum()} intermediate)")
