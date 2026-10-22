# NetworkEvolveLab Benchmark (draft)

This directory is the public-facing benchmark/software layer for the paper's
frozen experiments.  It is intentionally thin at present: the executable
registry lives in `src/network_evolve/benchmark_suite.py`, and the cached
graphs live in `data/benchmarks/`; this folder documents the contract and
provides a validation entry point for a future NeurIPS D&B / JOSS package.

## What is included

- 5 calibrated surrogate networks (Twitter 2020, SNAP Higgs surrogate, Reddit
  ConvoKit, DEBATE, Birdwatch/Community Notes);
- 1 real topology graph: SNAP Higgs (no stance labels; topology-derived camps);
- 1 real **human-labeled** stance graph: Political Blogosphere (PolBlogs,
  Adamic & Glance 2005; left/right `value` labels);
- frozen result JSONs and a SHA256 provenance manifest.

## Registry manifest

`benchmark/manifest.json` lists every cached graph with its SHA256, node/edge counts, source, and recorded raw-source hashes. Regenerate it with:

```bash
python scripts/build_benchmark_manifest.py
```

## Quick validation

```bash
python scripts/validate_benchmark.py
```

## Reproducing the real-stance check

```bash
python scripts/build_polblogs_caches.py --nodes 100 --seeds 0-9
python -m src.network_evolve.phase8_real_stance --seeds 10 --rounds 25 \
    --nodes 100 --out results/phase8_real_stance.json
```

## Contract

- Each dataset loader returns `(networkx.DiGraph, ideology: dict[int, float])`.
- Node ids are `0..n-1`; ideology is in `[-1, 1]`.
- Real graphs document source URL, SHA256, license/citation, and the exact
  derivation recipe in `data/raw/README.md`.
- Cached graphs are JSON (`edges`, `ideology`, `metadata`) and are covered by
  `results/PROVENANCE.json`.

## Status / next additions

The current public benchmark has **2 real graphs** (one with human stance
labels).  A NeurIPS D&B submission would benefit from 1-2 additional real
platform graphs with stance labels and a per-task leaderboard; those are
deliberately not fabricated here.  JOSS-style packaging has the `network-evolve` entry point, MIT license, a
manifest check, and a CI validation step; it still needs narrated API docs
and 1-2 more real graphs before a D&B submission.
