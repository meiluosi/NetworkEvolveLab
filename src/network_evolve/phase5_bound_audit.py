"""Phase 5：模块度上界的一致性审计（Generalised Modularity Bound Audit）。

理论更正（对原论文 Proposition 1 的推广）
--------------------------------------
原论文只用 K=2 的特例（Q ≤ 1 − 1/K = 0.500）来「证伪」预注册假说 H1（Q > 0.65），
但该结论被表述为「任何双极拓扑下 Q > 0.65 在数学上不可能」。这个表述过强，
因为论文自身表格里的 Q 值高达 0.735（例如 Reddit ConvoKit 上 HK 模型），
与「Q ≤ 0.500」的绝对上限直接矛盾 —— 矛盾的根源是**把 K=2 的界当成了普适界**。

一般化（由 Cauchy–Schwarz 直接给出，无需等量假设）：
    对任意划分为 K 个社群的划分，新曼模块度满足
        Q = Σ_c [ e_cc/m − (vol_c/2m)² ] ≤ 1 − Σ_c a_c² ≤ 1 − 1/K,   a_c = vol_c/(2m), Σa_c = 1
    等号条件：各社群等量（a_c = 1/K）且无跨社群连边。
因此：
    · K = 2 → Q ≤ 0.500   （双极回音室，本项目演化的实际所在区间）
    · K = 3 → Q ≤ 0.667   （Q > 0.65 需要至少 3 个社群）
    · K = 4 → Q ≤ 0.750
即 H1 的 0.65 阈值**在原理上并非不可达**，只要网络碎裂成 ≥3 个社群即可；
真正被实验证伪的是「本引擎在全部参数网格下都不产生第三社群，因此始终停在双极区间」。

本模块做两件事：
  A. 对 SOTA 基准表的每一个 (数据集 × 方法) 单元，核验 Q ≤ 1 − 1/K，并给出 K 的分布，
     说明论文表格中的 Q > 0.5 的单元全部对应 K ≥ 3 的碎裂结构（与 Prop.1 不矛盾）。
  B. 对主演化引擎重跑 H1 参数网格，记录终态 (K, Q)，核验上界并给出
     「达到 Q > 0.65 所需的最少社群数」。

用法: python -m src.network_evolve.phase5_bound_audit --seeds 10 --out results/phase5_bound_audit.json
"""
import argparse
import json
import numpy as np
from multiprocessing import Pool

from .expkit import run_many, mean_ci, fmt
from .metrics import snapshot


def modularity_upper_bound(k: int) -> float:
    """K 个社群划分下的模块度上界 1 − 1/K。"""
    if k < 1:
        return float("nan")
    return 1.0 - 1.0 / float(k)


def _engine_run(args):
    """在主演化引擎上跑一轮并返回 (K, Q, 社群规模分布)。"""
    cfg, seed = args
    from .expkit import make_engine
    e = make_engine(cfg, seed)
    for _ in range(cfg.get("rounds", 60)):
        e.run_round()
    import networkx as nx
    und = e.graph.g.to_undirected()
    comms = nx.community.louvain_communities(und, weight="weight", seed=0)
    q = float(nx.community.modularity(und, comms, weight="weight")) if e.graph.g.number_of_edges() else 0.0
    sizes = sorted((len(c) for c in comms), reverse=True)
    return {"k": len(comms), "q": q, "sizes": sizes, "giant": snapshot(e)["giant_frac"],
            "cross": snapshot(e)["cross_edge_frac"]}


def main():
    ap = argparse.ArgumentParser(description="Phase 5: generalised modularity-bound consistency audit")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--out", default="results/phase5_bound_audit.json")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()

    out = {"args": vars(a), "theory": {
        "statement": "对 K 个社群的任意划分，Q <= 1 - 1/K（等号需等量且无跨社群边）",
        "bound_by_k": {str(k): modularity_upper_bound(k) for k in range(2, 9)},
        "min_communities_for_q_above_0.65": 3,
        "note": "K=2 -> 0.500；K=3 -> 0.667；K=4 -> 0.750。故 Q>0.65 需 K>=3，"
                "原论文以 K=2 特例断言『Q>0.65 数学上不可能』属过度推广。",
    }}

    # ——— A. 核验 SOTA 基准表 ———
    bench_path = "results/sota_benchmark.json"
    if __import__("os").path.exists(bench_path):
        bench = json.load(open(bench_path))["benchmark"]
        checks, violations = [], []
        for dname, rows in bench.items():
            for mname, row in rows.items():
                if "num_comms" not in row or row["num_comms"] == "N/A":
                    continue
                q = float(str(row["modularity_q"]).split("±")[0])
                k = int(float(str(row["num_comms"]).split("±")[0]))
                bound = modularity_upper_bound(k)
                ok = bool(q <= bound + 1e-9)
                checks.append({"dataset": dname, "model": mname, "Q": q, "K": k,
                               "upper_bound": bound, "holds": ok})
                if not ok:
                    violations.append(checks[-1])
        out["benchmark_bound_check"] = {
            "n_cells_checked": len(checks),
            "n_violations": len(violations),
            "violations": violations,
            "cells": checks,
            "note": "Q>0.5 的单元必然对应 K>=3 的碎裂结构，与 K=2 特例不矛盾。",
        }
        k_dist = {}
        for c in checks:
            k_dist[str(c["K"])] = k_dist.get(str(c["K"]), 0) + 1
        out["benchmark_bound_check"]["K_distribution"] = k_dist

    # ——— B. 主演化引擎 H1 参数网格的 (K, Q) ———
    grid = []
    seeds = list(range(a.seeds))
    for th in (0.2, 0.3, 0.4):
        for blk in (None, 1.2):
            for pf in (0.1, 0.3):
                cfg = dict(theta=th, block=blk, p_form=pf, rounds=a.rounds)
                jobs = [(cfg, s) for s in seeds]
                with Pool(a.procs) as pool:
                    runs = pool.map(_engine_run, jobs)
                qs = np.array([r["q"] for r in runs])
                ks = np.array([r["k"] for r in runs])
                bounds = np.array([modularity_upper_bound(int(k)) for k in ks])
                grid.append({
                    "config": f"theta={th}|block={blk}|p_form={pf}",
                    "Q_mean": float(qs.mean()), "Q_max": float(qs.max()),
                    "K_mean": float(ks.mean()), "K_max": int(ks.max()),
                    "K_min": int(ks.min()),
                    "frac_seeds_bipolar_K2": float(np.mean(ks == 2)),
                    "bound_max": float(bounds.max()),
                    "all_cells_within_bound": bool(np.all(qs <= bounds + 1e-9)),
                    "h1_threshold_0.65_reached": bool(np.any(qs > 0.65)),
                    "homo_comms_max": int(max(ks)),
                })
    out["engine_grid"] = grid

    n_grid = len(grid)
    out["summary"] = {
        "n_grid_configs": n_grid,
        "all_grid_cells_within_bound": all(g["all_cells_within_bound"] for g in grid),
        "any_grid_config_reaching_q_above_0.65": any(g["h1_threshold_0.65_reached"] for g in grid),
        "max_K_observed_in_engine": int(max(g["K_max"] for g in grid)),
        "engine_stays_bipolar": all(g["frac_seeds_bipolar_K2"] > 0.5 for g in grid),
        "benchmark_violations": out.get("benchmark_bound_check", {}).get("n_violations", "n/a"),
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print(f"[Done] Phase 5 bound audit written to {a.out}")


if __name__ == "__main__":
    main()
