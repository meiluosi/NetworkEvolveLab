"""Public stance-labeled real graph: the 2005 Political Blogosphere (PolBlogs).

Source
------
Lada A. Adamic and Natalie Glance, "The political blogosphere and the 2004 US
Election", *Proceedings of the WWW-2005 Workshop on the Weblogging Ecosystem*.
The canonical GML file is distributed by Mark Newman at
``http://www-personal.umich.edu/~mejn/netdata/polblogs.zip``.

The graph contains 1,490 political blogs.  Each node carries a ``value`` field:
0 = left/liberal, 1 = right/conservative, derived from blog directories and
manual checks by the original authors.  Edges are directed blog-to-blog links
extracted from front pages in February 2005.  This is *human-annotated* stance,
unlike the topology-derived camps used for the SNAP Higgs subgraph.

Construction used here
-----------------------
For each seed we take the top-``4*n`` nodes by total degree, sample ``n`` of
them with a seeded RNG, induce the subgraph, re-index nodes in ascending
original id, and assign the binary stance label as a belief level:
left = -0.7, right = +0.7, plus a seeded N(0, 0.05) jitter (the jitter only
makes the continuous dynamics non-degenerate; the label is the original
``value``).  Edge weight is the constant 0.6 used for the other real-data
cache, because the released GML has direction but no interaction strength.

Cache format matches ``datasets.py`` / ``datasets_real.py``::

    {"edges": [[u, v, w], ...],
     "ideology": {"0": -0.7, ...},
     "metadata": {...}}
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "raw")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "benchmarks")

POLBLOGS_FILENAME = "polblogs.gml"
POLBLOGS_URL = "http://www-personal.umich.edu/~mejn/netdata/polblogs.zip"
POLBLOGS_CITATION = ("Adamic, L. A.; and Glance, N. 2005. The political "
                     "blogosphere and the 2004 US election. In Proc. WWW-2005 "
                     "Workshop on the Weblogging Ecosystem.")
POLBLOGS_SHA256 = "bd83e3dd2397a57d2252cc212524c21fee6708663136898d4ace523224bdc632"
EDGE_WEIGHT = 0.6
LABEL_MAGNITUDE = 0.7
JITTER_STD = 0.05


def polblogs_raw_path() -> Optional[str]:
    p = os.path.join(RAW_DIR, POLBLOGS_FILENAME)
    return p if os.path.exists(p) else None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_raw() -> nx.DiGraph:
    """Parse the GML manually: the released file contains duplicate edges, which
    ``networkx.read_gml`` rejects.  Node ids are integers; edges are deduplicated.
    """
    path = polblogs_raw_path()
    if path is None:
        raise FileNotFoundError(
            f"{POLBLOGS_FILENAME} not found under data/raw/; download from {POLBLOGS_URL}")
    text = open(path, encoding="utf-8").read()
    node_re = re.compile(r"node\s*\[(.*?)\]", re.S)
    edge_re = re.compile(r"edge\s*\[(.*?)\]", re.S)
    values = {}
    for block in node_re.findall(text):
        m_id = re.search(r"\bid\s+(\d+)", block)
        m_v = re.search(r"\bvalue\s+([01])\b", block)
        if m_id and m_v:
            values[int(m_id.group(1))] = int(m_v.group(1))
    g = nx.DiGraph()
    g.add_nodes_from(sorted(values))
    seen = set()
    for block in edge_re.findall(text):
        m_s = re.search(r"\bsource\s+(\d+)", block)
        m_t = re.search(r"\btarget\s+(\d+)", block)
        if not (m_s and m_t):
            continue
        u, v = int(m_s.group(1)), int(m_t.group(1))
        if u in values and v in values and u != v and (u, v) not in seen:
            seen.add((u, v))
            g.add_edge(u, v)
    for n, value in values.items():
        g.nodes[n]["value"] = value
    return g


def build_polblogs_cache(num_nodes: int, seed: int, cache_dir: str = CACHE_DIR) -> str:
    raw = _load_raw()
    value = {n: int(d["value"]) for n, d in raw.nodes(data=True)}
    deg = dict(raw.degree())
    pool_size = min(len(deg), max(num_nodes, int(4 * num_nodes)))
    pool = sorted(deg, key=lambda n: (-deg[n], n))[:pool_size]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=min(num_nodes, len(pool)), replace=False)
    chosen = sorted(int(n) for n in chosen)  # original ids; read_gml may use strings
    # networkx read_gml uses node labels as ids by default; map chosen back by identity.
    chosen_nodes = chosen
    sub = raw.subgraph(chosen_nodes).copy()
    mapping = {old: i for i, old in enumerate(chosen_nodes)}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(chosen_nodes)))
    for u, v in sub.edges():
        if u in mapping and v in mapping:
            g.add_edge(mapping[u], mapping[v], weight=EDGE_WEIGHT)
    jitter = np.random.default_rng(seed + 991).normal(0.0, JITTER_STD, size=len(chosen_nodes))
    ideology = {}
    for old, i in mapping.items():
        sign = -1.0 if value[old] == 0 else 1.0
        ideology[str(i)] = float(np.clip(sign * LABEL_MAGNITUDE + jitter[i], -1.0, 1.0))
    payload = {
        "edges": [[int(u), int(v), float(EDGE_WEIGHT)] for u, v in g.edges()],
        "ideology": ideology,
        "metadata": {
            "source": "Political Blogosphere (PolBlogs), Feb 2005",
            "source_url": POLBLOGS_URL,
            "citation": POLBLOGS_CITATION,
            "source_sha256": POLBLOGS_SHA256,
            "selection": (f"top {pool_size} nodes by total degree; seeded sample of "
                          f"{len(chosen_nodes)}; induced subgraph; ids ascending"),
            "labels": "value 0=left/liberal -> -0.7; value 1=right/conservative -> +0.7; "
                      "seeded N(0,0.05) jitter only for continuous dynamics",
            "num_nodes": len(chosen_nodes),
            "num_edges": int(g.number_of_edges()),
            "seed": int(seed),
        },
    }
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"polblogs_real_n{num_nodes}_s{seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def build_polblogs_caches(num_nodes: int = 60, seeds=range(10),
                          cache_dir: str = CACHE_DIR) -> Dict[int, str]:
    return {s: build_polblogs_cache(num_nodes, s, cache_dir) for s in seeds}


def load_polblogs_real_subgraph(num_nodes: int, seed: int = 0):
    path = os.path.join(CACHE_DIR, f"polblogs_real_n{num_nodes}_s{seed}.json")
    if not os.path.exists(path):
        build_polblogs_cache(num_nodes, seed)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    g = nx.DiGraph()
    n = int(d["metadata"]["num_nodes"])
    g.add_nodes_from(range(n))
    for u, v, w in d["edges"]:
        g.add_edge(int(u), int(v), weight=float(w))
    ideology = {int(k): float(v) for k, v in d["ideology"].items()}
    return g, ideology


def polblogs_stats(num_nodes: int, seed: int = 0) -> Dict[str, float]:
    g, ideology = load_polblogs_real_subgraph(num_nodes, seed)
    und = g.to_undirected()
    comms = nx.community.louvain_communities(und, seed=seed) if und.number_of_edges() else []
    q = nx.community.modularity(und, comms) if comms else 0.0
    left = sum(1 for v in ideology.values() if v < 0)
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "density": float(nx.density(g)),
        "reciprocity": float(nx.overall_reciprocity(g)) if g.number_of_edges() else 0.0,
        "num_communities": len(comms),
        "modularity_q": float(q),
        "left_frac": left / max(1, len(ideology)),
        "right_frac": 1.0 - left / max(1, len(ideology)),
    }
