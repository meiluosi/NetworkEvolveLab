"""Degree- and reciprocity-matched topology variants (Phase 1c robustness).

The four Phase-1c topologies differ not only in mesoscopic structure but also in
their initial mean out-degree and reciprocity (ER: low reciprocity, WS/BA:
~0.75, empirical-bipolar: 1.0).  This module builds a *matched* variant of each
topology by starting from its natural undirected skeleton and then:

1. adding/removing random edges until the skeleton has the target edge count;
2. orienting each skeleton edge, making exactly ``q`` of them mutual, so that
   the final directed graph has the requested mean out-degree and edge-level
   reciprocity.

The skeleton family (ER / BA / WS / two-camp empirical-bipolar) is preserved;
only the edge budget and the reciprocity are matched.  The achieved mean degree
and reciprocity are returned alongside the graph for auditing.
"""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .graph import DynamicSocialGraph


def _skeleton(topology: str, num_nodes: int, seed: int) -> nx.Graph:
    g = DynamicSocialGraph(num_nodes=num_nodes, seed=seed,
                           topology_type=topology, initial_density=0.08).g
    s = nx.Graph()
    s.add_nodes_from(range(num_nodes))
    for u, v in g.edges():
        if u != v:
            s.add_edge(u, v)
    return s


def _resize_edges(s: nx.Graph, target_m: int, rng: np.random.Generator) -> nx.Graph:
    """Add/remove edges to reach target_m while keeping the graph connected."""
    edges = list(s.edges())
    rng.shuffle(edges)
    if s.number_of_edges() > target_m:
        for u, v in edges:
            if s.number_of_edges() <= target_m:
                break
            if not s.has_edge(u, v):
                continue
            s.remove_edge(u, v)
            if not nx.is_connected(s):
                s.add_edge(u, v)
    elif s.number_of_edges() < target_m:
        all_pairs = [(u, v) for u in range(s.number_of_nodes())
                     for v in range(u + 1, s.number_of_nodes())]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            if s.number_of_edges() >= target_m:
                break
            if not s.has_edge(u, v):
                s.add_edge(u, v)
    return s


def build_matched_graph(topology: str, num_nodes: int, seed: int,
                        target_degree: float, target_reciprocity: float
                        ) -> Tuple[nx.DiGraph, Dict[str, float]]:
    """Build a directed graph with requested mean out-degree and reciprocity.

    Reciprocity is measured as ``nx.overall_reciprocity``: the fraction of
    directed edges whose reverse is also present.  For a skeleton with ``M``
    undirected edges and ``q`` mutual edges, the directed edge count is
    ``M + q`` and reciprocity is ``2q / (M + q)``; we solve for ``q`` from the
    target mean out-degree and then pick the closest integer mutual count.
    """
    rng = np.random.default_rng(seed)
    p_mutual = target_reciprocity / (2.0 - target_reciprocity)
    target_edges = int(round(target_degree * num_nodes))
    target_m = int(round(target_edges / (1.0 + p_mutual)))
    target_m = max(num_nodes - 1, min(target_m, num_nodes * (num_nodes - 1) // 2))
    target_q = max(0, min(target_m, target_edges - target_m))

    s = _skeleton(topology, num_nodes, seed)
    s = _resize_edges(s, target_m, rng)

    edges = list(s.edges())
    rng.shuffle(edges)
    mutual = set(edges[:target_q])
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    for u, v in edges:
        w = float(rng.uniform(0.3, 0.9))
        if (u, v) in mutual:
            g.add_edge(u, v, weight=w)
            g.add_edge(v, u, weight=w)
        else:
            if rng.random() < 0.5:
                u, v = v, u
            g.add_edge(u, v, weight=w)

    mean_out = g.number_of_edges() / num_nodes
    stats = {
        "target_degree": float(target_degree),
        "target_reciprocity": float(target_reciprocity),
        "mean_out_degree": float(mean_out),
        "reciprocity": float(nx.overall_reciprocity(g)),
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_mutual_edges": int(target_q),
    }
    return g, stats
