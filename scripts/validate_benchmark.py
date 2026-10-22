#!/usr/bin/env python3
"""Validate the benchmark dataset registry and cached real/stance graphs.

Checks that every registered synthetic dataset loads, that the cached real
graphs (SNAP Higgs, PolBlogs) load with the expected node count, and that
stance labels are finite and in [-1, 1].  Exits non-zero on any failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.network_evolve.benchmark_suite import DATASETS, get_dataset
from src.network_evolve.datasets_stance import load_polblogs_real_subgraph
from src.network_evolve.datasets_polbooks import load_polbooks_real_subgraph


def check_dataset(name: str, n: int = 60):
    g, ideo = get_dataset(name, n, 0)
    assert g.number_of_nodes() == n, (name, g.number_of_nodes(), n)
    assert g.number_of_edges() > 0, name
    vals = np.array(list(ideo.values()), dtype=float)
    assert vals.size == n and np.isfinite(vals).all(), name
    assert vals.min() >= -1.01 and vals.max() <= 1.01, name
    return g.number_of_edges(), float(vals.mean())


def main():
    failures = []
    for name in DATASETS:
        try:
            edges, mean_label = check_dataset(name, 60)
            print(f"[ok] {name:<24} n=60 edges={edges} label_mean={mean_label:+.3f}")
        except Exception as exc:  # noqa: BLE001 - validation reports all failures
            failures.append((name, repr(exc)))
            print(f"[FAIL] {name}: {exc!r}")
    try:
        g, ideo = load_polbooks_real_subgraph(100, 0)
        assert g.number_of_nodes() == 100 and g.number_of_edges() > 0
        vals = np.array(list(ideo.values()), dtype=float)
        assert np.isfinite(vals).all() and vals.min() >= -1 and vals.max() <= 1
        neg = float(np.mean(vals < 0))
        print(f"[ok] {'PolBooks_Real':<24} n=100 edges={g.number_of_edges()} "
              f"neg_frac={neg:.2f}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("PolBooks_Real", repr(exc)))
        print(f"[FAIL] PolBooks_Real: {exc!r}")
    try:
        g, ideo = load_polblogs_real_subgraph(100, 0)
        assert g.number_of_nodes() == 100 and g.number_of_edges() > 0
        vals = np.array(list(ideo.values()), dtype=float)
        assert np.isfinite(vals).all() and vals.min() >= -1 and vals.max() <= 1
        left = float(np.mean(vals < 0))
        print(f"[ok] {'PolBlogs_Real':<24} n=100 edges={g.number_of_edges()} "
              f"left_frac={left:.2f}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("PolBlogs_Real", repr(exc)))
        print(f"[FAIL] PolBlogs_Real: {exc!r}")
    if failures:
        print(f"\n[FAIL] {len(failures)} dataset(s) failed validation")
        return 1
    print("\n[PASS] benchmark registry and cached real graphs validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
