#!/usr/bin/env python3
"""Write benchmark/manifest.json: dataset registry, caches, and source hashes."""
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


def main():
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
                         ("polblogs.txt", "PolBlogs metadata")):
        path = os.path.join(ROOT, "data", "raw", fname)
        if os.path.exists(path):
            raw.append({"file": f"data/raw/{fname}", "label": label, "sha256": sha(path)})
    manifest = {
        "name": "NetworkEvolveLab benchmark",
        "version": "0.1",
        "description": "5 calibrated surrogates + real SNAP Higgs topology + "
                       "PolBlogs human-labeled stance graph",
        "real_graphs": 2,
        "human_labeled_stance_graphs": 1,
        "caches": caches,
        "raw_sources": raw,
        "provenance": "results/PROVENANCE.json",
        "note": "Not yet a D&B/JOSS-ready leaderboard: 2 real graphs, no hidden test split. "
                "Add 1-2 more real stance graphs and a submission format before D&B.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[write] {OUT} ({len(caches)} caches, {len(raw)} raw sources)")


if __name__ == "__main__":
    main()
