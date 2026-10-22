#!/usr/bin/env python3
"""Build cached PolBooks human-labeled stance subgraphs."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.network_evolve.datasets_polbooks import (  # noqa: E402
    RAW_SHA256, build_caches, raw_path,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--seeds", default="0-9")
    a = ap.parse_args()
    print("raw file:", raw_path()); print("sha256  :", RAW_SHA256)
    seeds = []
    for part in a.seeds.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-"); seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    build_caches(a.nodes, seeds)
    print(f"[done] {len(seeds)} PolBooks caches written")


if __name__ == "__main__":
    main()
