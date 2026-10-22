#!/usr/bin/env python3
"""Shard Phase 6 by seed, run shards concurrently, and merge the results.

Why: a single Phase-6 process is sequential and each MiniMax batch call takes
~1-2 minutes.  Sharding by seed lets independent seeds run concurrently while
keeping each shard's JSON self-contained.  The merge recomputes the arm
summaries and paired contrasts from the per-seed rows; it never invents values.

Usage:
    python scripts/run_phase6_parallel.py --seeds 10 --jobs 10 \
        --llm-events 36 --model MiniMax-M3.1-Flash \
        --out results/phase6_highcov_36_m3_1.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from src.network_evolve.expkit import mean_ci  # noqa: E402

METRICS = ["modularity_q", "polarization", "giant_frac", "small_comm_frac",
           "rupture_asymmetry", "num_comms", "num_edges",
           "bridger_retention_mean", "other_retention_mean", "n_interactions"]
ARMS = ["rule", "llm_hybrid", "placebo"]


def _contrast(per_seed, a_arm, b_arm, key):
    a = np.array([r[a_arm][key] for r in per_seed], dtype=float)
    b = np.array([r[b_arm][key] for r in per_seed], dtype=float)
    d = a - b
    mean, half = mean_ci(d)
    if np.allclose(d, d[0]):
        p = 1.0 if d[0] == 0 else 0.0
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    return {"mean_diff_llm_minus_rule": float(mean),
            "mean_diff": float(mean),
            "ci95": [float(half), -float(half)],
            "p_value_paired_t": p,
            "n_positive": int(np.sum(d > 0)), "n_pairs": int(d.size),
            "per_seed_diff": [float(x) for x in d]}


def _merge_rule_agreement(rows):
    out = {"n_compared": 0, "n_agree": 0, "confusion": {},
           "llm_action_counts": {}, "rule_action_counts": {}}
    for r in rows:
        ag = (r.get("llm_stats") or {}).get("rule_agreement") or {}
        out["n_compared"] += int(ag.get("n_compared", 0))
        out["n_agree"] += int(ag.get("n_agree", 0))
        for k, c in (ag.get("confusion") or {}).items():
            out["confusion"].setdefault(k, {})
            for k2, c2 in c.items():
                out["confusion"][k][k2] = out["confusion"][k].get(k2, 0) + int(c2)
        for k, c in (ag.get("llm_action_counts") or {}).items():
            out["llm_action_counts"][k] = out["llm_action_counts"].get(k, 0) + int(c)
        for k, c in (ag.get("rule_action_counts") or {}).items():
            out["rule_action_counts"][k] = out["rule_action_counts"].get(k, 0) + int(c)
    return out


def merge(shards, args):
    per_seed = []
    for s in sorted(shards):
        d = json.load(open(shards[s], encoding="utf-8"))
        row = d["per_seed"][str(s)]
        per_seed.append(row)
    out = {"args": vars(args), "merge_note": "sharded by seed; merged without recomputation of raw events",
           "design": {"arms": ARMS, "pairing": "same seed initial conditions",
                      "statistical_unit": "seed", "shards": len(per_seed)},
           "per_seed": {str(i): r for i, r in zip(sorted(shards), per_seed)}}
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for key in METRICS:
            arr = np.array([r[arm][key] for r in per_seed], dtype=float)
            mean, half = mean_ci(arr)
            arms[arm][key] = {"mean": float(mean), "ci95": float(half),
                              "per_seed": [float(x) for x in arr]}
    out["arms"] = arms
    out["contrasts"] = {}
    for a_arm, b_arm, tag in (("llm_hybrid", "rule", "llm_minus_rule"),
                              ("placebo", "rule", "placebo_vs_rule"),
                              ("llm_hybrid", "placebo", "llm_vs_placebo")):
        out["contrasts"][tag] = {k: _contrast(per_seed, a_arm, b_arm, k)
                                 for k in ("modularity_q", "polarization",
                                           "giant_frac", "small_comm_frac",
                                           "rupture_asymmetry", "num_comms",
                                           "num_edges", "n_interactions")}
    # flatten llm-vs-rule contrast to the legacy keys used by the table builder
    flat = {}
    for k, v in out["contrasts"]["llm_minus_rule"].items():
        flat[k] = {"mean_diff_llm_minus_rule": v["mean_diff_llm_minus_rule"],
                   "ci95": v["ci95"], "p_value_paired_t": v["p_value_paired_t"],
                   "n_positive": v["n_positive"], "n_pairs": v["n_pairs"],
                   "per_seed_diff": v["per_seed_diff"]}
    out["contrasts"].update(flat)

    totals = {"n_events": 0, "n_valid": 0, "n_invalid": 0, "n_calls": 0,
              "n_cache_hits": 0, "n_rounds_with_llm": 0, "n_applied": 0,
              "n_interactions": 0, "actions": {}}
    for r in per_seed:
        st = r.get("llm_stats") or {}
        for k in ("n_events", "n_valid", "n_invalid", "n_calls", "n_cache_hits",
                  "n_rounds_with_llm", "n_applied", "n_interactions"):
            totals[k] += int(st.get(k, 0))
        for a, c in (st.get("actions") or {}).items():
            totals["actions"][a] = totals["actions"].get(a, 0) + int(c)
    out["llm_stats_total"] = totals
    out["llm_coverage"] = {
        "llm_asked_events": totals["n_events"],
        "llm_decided_events": totals["n_applied"],
        "engine_interactions_llm_arm": totals["n_interactions"],
        "coverage_of_interactions": (totals["n_applied"] / totals["n_interactions"]
                                     if totals["n_interactions"] else 0.0),
        "parse_success_rate": (totals["n_valid"] / totals["n_events"]
                               if totals["n_events"] else 0.0),
        "action_distribution": totals["actions"],
        "rule_agreement": _merge_rule_agreement(per_seed),
    }
    # add raw agreement / kappa at merge level
    ag = out["llm_coverage"]["rule_agreement"]
    n = int(ag["n_compared"])
    actions = ("reinforce", "decay", "block")
    conf = ag["confusion"]
    trace = sum((conf.get(a) or {}).get(a, 0) for a in actions)
    row_tot = {r: sum((conf.get(r) or {}).values()) for r in actions}
    col_tot = {c: sum((conf.get(r) or {}).get(c, 0) for r in actions) for c in actions}
    p_o = trace / n if n else 0.0
    p_e = (sum(row_tot[a] * col_tot[a] for a in actions) / (n * n)) if n else 0.0
    ag["agreement_rate"] = p_o
    ag["cohen_kappa"] = (p_o - p_e) / (1 - p_e) if n and p_e < 1 else 0.0
    ag["rule_block_events"] = int(col_tot.get("block", 0))
    ag["rule_decay_events"] = int(col_tot.get("decay", 0))
    ag["rule_severe_events"] = ag["rule_block_events"] + ag["rule_decay_events"]
    ag["severe_agreement"] = int((conf.get("block") or {}).get("block", 0)
                                 + (conf.get("decay") or {}).get("decay", 0))
    ag["severe_agreement_rate"] = (ag["severe_agreement"] / ag["rule_severe_events"]
                                   if ag["rule_severe_events"] else 0.0)
    out["llm_ledger"] = {
        "api_calls": sum(int((r.get("llm_stats") or {}).get("n_calls", 0)) for r in per_seed),
        "cache_hits": sum(int((r.get("llm_stats") or {}).get("n_cache_hits", 0)) for r in per_seed),
        "shards": len(per_seed),
        "note": "per-seed shards; each shard's calls are distinct API requests",
    }
    out["placebo"] = {
        "n_unfollows_per_seed": [float((r.get("placebo_stats") or {}).get("placebo", {}).get("n_unfollows", 0))
                                 for r in per_seed],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--llm-events", type=int, default=36)
    ap.add_argument("--model", default="MiniMax-M3.1-Flash")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking-budget", type=int, default=512)
    ap.add_argument("--budget-calls", type=int, default=400)
    ap.add_argument("--out", default="results/phase6_highcov_36_m3_1.json")
    ap.add_argument("--shard-dir", default=None)
    a = ap.parse_args()
    shard_dir = a.shard_dir or (os.path.splitext(a.out)[0] + "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    env = dict(os.environ)

    running = []
    shards = {}
    pending = list(range(a.seeds))
    while pending or running:
        while pending and len(running) < a.jobs:
            s = pending.pop(0)
            shard_path = os.path.join(shard_dir, f"seed{s}.json")
            log_path = os.path.join(shard_dir, f"seed{s}.log")
            cmd = [sys.executable, "-m", "src.network_evolve.phase6_llm_engine",
                   "--seeds", "1", "--seed-start", str(s), "--rounds", str(a.rounds),
                   "--nodes", str(a.nodes), "--llm-events", str(a.llm_events),
                   "--model", a.model, "--max-tokens", str(a.max_tokens),
                   "--thinking-budget", str(a.thinking_budget),
                   "--budget-calls", str(a.budget_calls), "--out", shard_path]
            log = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=os.path.abspath("."), env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((s, proc, log, shard_path))
            print(f"[start] seed {s} -> {shard_path}", flush=True)
        time.sleep(5)
        for item in list(running):
            s, proc, log, shard_path = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                if proc.returncode == 0:
                    shards[s] = shard_path
                    print(f"[done] seed {s}", flush=True)
                else:
                    print(f"[FAIL] seed {s} rc={proc.returncode}; see {shard_dir}/seed{s}.log",
                          flush=True)
        if not pending and not running:
            break
    missing = [s for s in range(a.seeds) if s not in shards]
    merged = merge(shards, a)
    if missing:
        merged["missing_seeds"] = missing
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[merged] {len(shards)}/{a.seeds} seeds -> {a.out}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
