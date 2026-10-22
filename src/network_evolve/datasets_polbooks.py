"""Public political co-purchase network with human stance labels (PolBooks).

Source
------
Valdis Krebs / Mark Newman, "Books about US politics" (2006):
105 books sold by Amazon, edges = frequent co-purchasing, and each node has a
human-assigned label ``l`` (liberal), ``n`` (neutral), or ``c`` (conservative).
The labels were assigned by Mark Newman from book descriptions and reviews, so
this is a genuine human-labeled stance graph (unlike topology-derived camps).

Derivation used here
--------------------
The graph has only 105 nodes.  For each seed we sample 100 nodes (seeded),
induce the co-purchase subgraph, re-index nodes in ascending original id, map
``c`` -> +0.7, ``l`` -> -0.7, ``n`` -> 0.0, and use a constant edge weight 0.6.
Cache format matches the rest of the benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "benchmarks")
RAW_FILENAME = "polbooks.gml"
RAW_SHA256 = "29e2439036fa4e13b478a9d4ceda78b5ab4d4d8252882a7f2b6036b028e34848"
RAW_URL = "http://www-personal.umich.edu/~mejn/netdata/polbooks.zip"
CITATION = ("V. Krebs / M. Newman, Books about US politics (2006); "
            "labels assigned by M. Newman; see orgnet.com.")
EDGE_WEIGHT = 0.6
LABEL_MAP = {"l": -0.7, "n": 0.0, "c": 0.7}


def raw_path() -> Optional[str]:
    p = os.path.join(RAW_DIR, RAW_FILENAME)
    return p if os.path.exists(p) else None


def build_cache(num_nodes: int, seed: int, cache_dir: str = CACHE_DIR) -> str:
    p = raw_path()
    if p is None:
        raise FileNotFoundError(f"{RAW_FILENAME} missing; download {RAW_URL}")
    raw = nx.read_gml(p)
    labels = {n: str(d.get("value", "n")) for n, d in raw.nodes(data=True)}
    nodes = sorted(raw.nodes())
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(nodes, size=min(num_nodes, len(nodes)), replace=False).tolist())
    cmap = {u: i for i, u in enumerate(chosen)}
    sub = raw.subgraph(chosen)
    edges = sorted({(min(cmap[u], cmap[v]), max(cmap[u], cmap[v]))
                    for u, v in sub.edges() if u in cmap and v in cmap})
    payload = {
        "edges": [[int(u), int(v), EDGE_WEIGHT] for u, v in edges],
        "ideology": {str(cmap[u]): float(LABEL_MAP.get(labels[u], 0.0)) for u in chosen},
        "metadata": {
            "source": "Books about US politics (PolBooks), 2006",
            "source_url": RAW_URL,
            "source_sha256": RAW_SHA256,
            "citation": CITATION,
            "labels": "human-assigned l/n/c -> -0.7/0.0/+0.7",
            "selection": f"seeded sample of {len(chosen)} of 105 book nodes; induced co-purchase subgraph",
            "num_nodes": len(chosen),
            "num_edges": len(edges),
            "seed": int(seed),
        },
    }
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"polbooks_real_n{num_nodes}_s{seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def build_caches(num_nodes: int = 100, seeds=range(10), cache_dir: str = CACHE_DIR):
    return {s: build_cache(num_nodes, s, cache_dir) for s in seeds}


def load_polbooks_real_subgraph(num_nodes: int, seed: int = 0):
    path = os.path.join(CACHE_DIR, f"polbooks_real_n{num_nodes}_s{seed}.json")
    if not os.path.exists(path):
        build_cache(num_nodes, seed)
    d = json.load(open(path, encoding="utf-8"))
    g = nx.DiGraph()
    g.add_nodes_from(range(int(d["metadata"]["num_nodes"])))
    for u, v, w in d["edges"]:
        # The source graph is undirected; represent each co-purchase tie as a
        # reciprocal directed edge so the same DiGraph interface is preserved.
        g.add_edge(int(u), int(v), weight=float(w))
        g.add_edge(int(v), int(u), weight=float(w))
    ideology = {int(k): float(v) for k, v in d["ideology"].items()}
    return g, ideology
