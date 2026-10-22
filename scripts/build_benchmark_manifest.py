#!/usr/bin/env python3
"""Write benchmark/manifest.json: dataset registry, caches, and source hashes."""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

ROOT = os.path.abspath(".")
BENCH = os.path.join(ROOT, "data", "benchmarks")
OUT = os.path.join(ROOT, "benchmark", "manifest.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_manifest():
    caches = []
    for fname in sorted(os.listdir(BENCH)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(BENCH, fname)
        d = json.load(open(path, encoding="utf-8"))
        if not isinstance(d, dict):
            continue
        caches.append({
            "cache": f"data/benchmarks/{fname}",
            "sha256": sha(path),
            "num_nodes": len(d.get("ideology", {})),
            "num_edges": len(d.get("edges", [])),
            "source": (d.get("metadata") or {}).get("source", "synthetic loader"),
            "license": (d.get("metadata") or {}).get("license", "see data/raw/README.md"),
        })
    raw = []
    for fname, label in (("polblogs.gml", "PolBlogs (human stance labels)"),
                         ("polblogs.txt", "PolBlogs metadata"),
                         ("polbooks.gml", "PolBooks (human stance labels)"),
                         ("polbooks.txt", "PolBooks metadata")):
        path = os.path.join(ROOT, "data", "raw", fname)
        if os.path.exists(path):
            raw.append({"file": f"data/raw/{fname}", "label": label, "sha256": sha(path)})
    manifest = {
        "name": "NetworkEvolveLab benchmark",
        "version": "0.1",
        "description": "5 calibrated surrogates + real SNAP Higgs topology + "
                       "PolBlogs and PolBooks human-labeled stance graphs",
        "real_graphs": 3,
        "human_labeled_stance_graphs": 2,
        "caches": caches,
        "raw_sources": raw,
        "provenance": "results/PROVENANCE.json",
        "note": "Not yet a D&B/JOSS-ready leaderboard: 3 real graphs, no hidden test split. "
                "Add 1-2 more real platform graphs and a submission format before D&B.",
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify benchmark/manifest.json is current without writing")
    a = ap.parse_args()
    manifest = build_manifest()
    if a.check:
        if not os.path.exists(OUT):
            print(f"[FAIL] missing {OUT}")
            return 1
        old = json.load(open(OUT, encoding="utf-8"))
        if old != manifest:
            print(f"[STALE] {OUT}; run python scripts/build_benchmark_manifest.py")
            return 1
        print(f"[ok] {OUT} is current")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[write] {OUT} ({len(manifest['caches'])} caches, {len(manifest['raw_sources'])} raw sources)")
    return 0


if __name__ == "__main__":
    main()
