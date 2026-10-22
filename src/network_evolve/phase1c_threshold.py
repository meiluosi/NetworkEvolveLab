"""Phase 1c：跨拓扑相变临界点 θ* 的**可分辨**测量（Cross-Topology Critical Threshold）。

为什么需要这个实验
------------------
`phase_robustness.py` 的设计意图是检验「θ* 是否落在 0.50~0.55 区间」，但它使用的
θ 网格是 ``[0.35, 0.45, 0.55, 0.65]`` —— 该区间内**只有一个采样点（0.55）**，
且 0.50 根本没有被采样。这样的网格在原理上就无法判定被检验的命题，
而正文却据此写下了「四个拓扑的临界容忍度一致落在 θ* ∈ [0.50, 0.55]」。
这是典型的「实验设计无法支撑断言」（ARS Mode 1 / Mode 3）。

判据设计
--------
本模块同时给出三个**互不相同但各自有物理含义**的判据，并按种子逐个估计：

1. ``theta_frag``（主判据，碎裂—连通转变）：
   最小 θ 使得该种子的巨型连通分量占比 ``giant_frac >= 0.99``。
   低于该阈值网络因不宽容而**碎裂**（断边成孤岛），高于该阈值网络保持全局连通。
   这是「临界容忍度」最直接的含义：**再宽容一点就还能维持连通性**。
2. ``theta_q``（辅判据，模块度塌缩）：
   最小 θ 使得 ``Q_end(θ) <= 0.5 * Q_end(θ_min)``。
   注意：对小世界 / 实证双极网络，模块度存在**结构性下限**（残余社群结构不随
   共识而消失），该阈值可能始终不成立 —— 此时记为**右删失**并显式报告，
   绝不静默丢弃，更不会把「未跨过」当成「跨过了」。
3. ``theta_pol``（辅判据，极化塌缩）：最小 θ 使得极化跌破起点值的一半。

三个判据回答的是不同问题；跨拓扑一致性由数据判定，而不是由我们预先选定。
删失样本以网格上界计入，并报告删失比例。
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from .expkit import run_config, fmt, mean_ci

TOPOLOGIES = ["erdos_renyi", "scale_free", "small_world", "empirical_bipolar"]

# 预登记的待检验区间（原正文的断言）
PREREGISTERED_BAND = (0.50, 0.55)

# 巨型分量恢复阈值（全局连通性判据）
GIANT_THRESHOLD = 0.99


def _theta_grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _crossing(theta_grid, values, level, direction, interp: bool = True):
    """求 values 穿越 level 的最小 θ（线性插值）。

    ``direction='down'``：values 跌破 level（模度 / 极化塌缩）
    ``direction='up'``  ：values 升过 level（巨型分量恢复）

    返回 ``(theta_star, censored)``；``censored=True`` 表示在网格上界仍未穿越。
    """
    for i, v in enumerate(values):
        crossed = (v <= level) if direction == "down" else (v >= level)
        if crossed:
            if not interp or i == 0:
                return float(theta_grid[i]), False
            v0, v1 = values[i - 1], v
            t0, t1 = theta_grid[i - 1], theta_grid[i]
            if v0 == v1:
                return float(t1), False
            frac = (v0 - level) / (v0 - v1)
            return float(t0 + frac * (t1 - t0)), False
    return float(theta_grid[-1]), True


def _summarize(values, censored_flags, seeds_n):
    arr = np.asarray(values, dtype=float)
    m, h = mean_ci(arr)
    return {
        "per_seed": [float(x) for x in arr],
        "mean": float(m),
        "ci95": float(h),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_censored": int(sum(censored_flags)),
        "frac_censored": float(sum(censored_flags) / seeds_n) if seeds_n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Topology Critical Threshold Sweep")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--theta-min", type=float, default=0.15)
    parser.add_argument("--theta-max", type=float, default=0.85)
    parser.add_argument("--theta-step", type=float, default=0.05)
    parser.add_argument("--procs", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/phase1c_threshold.json")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    thetas = _theta_grid(args.theta_min, args.theta_max, args.theta_step)
    print(f"=== Phase 1c: 跨拓扑相变临界点测量 "
          f"(Seeds={args.seeds}, Rounds={args.rounds}, Nodes={args.nodes}) ===")
    print(f"θ grid: {thetas}")

    # —— 一次性并行跑完全部 (topology, θ, seed) 组合 ——
    jobs = []
    for topo in TOPOLOGIES:
        for th in thetas:
            cfg = dict(topology_type=topo, theta=th, rounds=args.rounds,
                       block=1.2, p_form=0.1, nodes=args.nodes)
            for s in seeds:
                jobs.append((cfg, s))
    with Pool(args.procs) as pool:
        raw = pool.map(run_config, jobs)

    # 重新组装为 topology → θ → seed → metrics
    per_topo = {topo: {th: [] for th in thetas} for topo in TOPOLOGIES}
    idx = 0
    for topo in TOPOLOGIES:
        for th in thetas:
            for _ in seeds:
                per_topo[topo][th].append(raw[idx])
                idx += 1

    results = {
        "args": vars(args),
        "theta_grid": thetas,
        "preregistered_band": list(PREREGISTERED_BAND),
        "giant_threshold": GIANT_THRESHOLD,
        "criteria": {
            "theta_frag": f"giant_frac >= {GIANT_THRESHOLD}（碎裂→连通），相邻点线性插值",
            "theta_q": "Q_end(θ) <= 0.5 * Q_end(θ_min)（模块度塌缩），相邻点线性插值",
            "theta_pol": "polarization(θ) <= 0.5 * polarization(θ_min)（极化塌缩），相邻点线性插值",
        },
        "topologies": {},
    }

    criterion_values = {"theta_frag": [], "theta_q": [], "theta_pol": []}
    for topo in TOPOLOGIES:
        # 逐种子曲线
        curves = {s: {"q": [], "pol": [], "giant": []} for s in seeds}
        for th in thetas:
            runs = per_topo[topo][th]
            for i, s in enumerate(seeds):
                curves[s]["q"].append(runs[i]["q_end"])
                curves[s]["pol"].append(runs[i]["polar"])
                curves[s]["giant"].append(runs[i]["giant"])

        curve = {}
        for th in thetas:
            runs = per_topo[topo][th]
            curve[str(th)] = {
                "q_end": fmt([r["q_end"] for r in runs]),
                "polarization": fmt([r["polar"] for r in runs]),
                "giant_frac": fmt([r["giant"] for r in runs]),
                "cross_edges": fmt([r["cross"] for r in runs]),
                "n_edges": fmt([r["edges"] for r in runs]),
            }

        frag, frag_c, qv, q_c, pv, p_c = [], [], [], [], [], []
        for s in seeds:
            q = curves[s]["q"]
            pol = curves[s]["pol"]
            g = curves[s]["giant"]
            t_f, c_f = _crossing(thetas, g, GIANT_THRESHOLD, "up")
            t_q, c_q = _crossing(thetas, q, 0.5 * q[0], "down")
            t_p, c_p = _crossing(thetas, pol, 0.5 * pol[0], "down")
            frag.append(t_f); frag_c.append(c_f)
            qv.append(t_q); q_c.append(c_q)
            pv.append(t_p); p_c.append(c_p)

        s_frag = _summarize(frag, frag_c, len(seeds))
        s_q = _summarize(qv, q_c, len(seeds))
        s_pol = _summarize(pv, p_c, len(seeds))
        lo, hi = PREREGISTERED_BAND
        results["topologies"][topo] = {
            "curve": curve,
            "theta_frag": s_frag,
            "theta_q": s_q,
            "theta_pol": s_pol,
            "in_preregistered_band_frag": bool(lo <= s_frag["mean"] <= hi),
            "in_preregistered_band_q": bool(lo <= s_q["mean"] <= hi),
        }
        for key, summ in (("theta_frag", s_frag), ("theta_q", s_q), ("theta_pol", s_pol)):
            criterion_values[key].extend(summ["per_seed"])
        print(f"  {topo:<20} θ*frag={s_frag['mean']:.3f}±{s_frag['ci95']:.3f} "
              f"(cens {s_frag['n_censored']}) | θ*Q={s_q['mean']:.3f}±{s_q['ci95']:.3f} "
              f"(cens {s_q['n_censored']}) | θ*Π={s_pol['mean']:.3f}±{s_pol['ci95']:.3f} "
              f"(cens {s_pol['n_censored']})")

    lo, hi = PREREGISTERED_BAND
    summary = {}
    for key, vals in criterion_values.items():
        arr = np.asarray(vals, dtype=float)
        n_in = int(((arr >= lo) & (arr <= hi)).sum())
        topo_means = [results["topologies"][t][key]["mean"] for t in TOPOLOGIES]
        summary[key] = {
            "n_seed_estimates": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "n_in_preregistered_band": n_in,
            "frac_in_preregistered_band": float(n_in / arr.size),
            "band_consistent": bool(n_in == arr.size),
            "per_topology_means": {
                t: results["topologies"][t][key]["mean"] for t in TOPOLOGIES},
            "per_topology_spread": float(max(topo_means) - min(topo_means)),
        }
        print(f"  [{key}] θ* ∈ [{arr.min():.3f}, {arr.max():.3f}] "
              f"(拓扑间极差 {summary[key]['per_topology_spread']:.3f}); "
              f"落在预登记区间 [{lo}, {hi}] 的估计 {n_in}/{arr.size} "
              f"= {summary[key]['frac_in_preregistered_band']:.0%}")
    results["summary"] = summary

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Done] Threshold sweep written to {args.out}")


if __name__ == "__main__":
    main()
