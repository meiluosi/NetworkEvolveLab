"""一次性构建真实 SNAP Higgs 基准子图缓存（10 个种子，单次扫描原始边表）。

用法: python scripts/build_real_higgs_caches.py --nodes 60 --seeds 0-9
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.network_evolve.datasets_real import (
    build_higgs_real_caches, higgs_raw_path, higgs_source_sha256, higgs_real_stats,
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

    raw = higgs_raw_path()
    print("raw file:", raw)
    print("sha256  :", higgs_source_sha256())
    if raw is None:
        print("原始文件不存在，无法构建缓存。")
        sys.exit(1)

    seeds = parse_seeds(a.seeds)
    t0 = time.time()
    written = build_higgs_real_caches(num_nodes=a.nodes, seeds=seeds, cache_dir=a.cache_dir)
    print(f"[done] {len(written)} 个缓存写入，用时 {time.time() - t0:.1f}s")

    for seed, path in written.items():
        st = higgs_real_stats(num_nodes=a.nodes, seed=seed)
        print(f"  seed {seed}: edges={st['num_edges']} density={st['density']:.3f} "
              f"reciprocity={st['reciprocity']:.3f} K={st['n_communities']} "
              f"alpha={st['power_law_alpha']:.2f} max_in_deg={st['max_in_degree']}")


if __name__ == "__main__":
    main()
