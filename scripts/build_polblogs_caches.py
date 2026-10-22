#!/usr/bin/env python3
"""Build cached PolBlogs real-stance subgraphs.

Usage: python scripts/build_polblogs_caches.py --nodes 60 --seeds 0-9
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.network_evolve.datasets_stance import (
    POLBLOGS_SHA256, build_polblogs_caches, polblogs_raw_path, polblogs_stats,
)


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--cache-dir", default="data/benchmarks")
    a = ap.parse_args()
    raw = polblogs_raw_path()
    print("raw file:", raw)
    print("sha256  :", POLBLOGS_SHA256)
    if raw is None:
        print("原始文件不存在，无法构建缓存。")
        sys.exit(1)
    t0 = time.time()
    written = build_polblogs_caches(num_nodes=a.nodes, seeds=parse_seeds(a.seeds),
                                    cache_dir=a.cache_dir)
    print(f"[done] {len(written)} 个缓存写入，用时 {time.time() - t0:.1f}s")
    for seed, path in written.items():
        st = polblogs_stats(num_nodes=a.nodes, seed=seed)
        print(f"  seed {seed}: edges={st['num_edges']} density={st['density']:.3f} "
              f"reciprocity={st['reciprocity']:.3f} K={st['num_communities']} "
              f"Q={st['modularity_q']:.3f} left={st['left_frac']:.2f}")


if __name__ == "__main__":
    main()
