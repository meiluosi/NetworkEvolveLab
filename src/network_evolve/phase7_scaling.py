"""Phase 7：网络规模敏感性研究（Network-Size Sensitivity of the Main Conclusions）。

为什么需要
----------
论文此前的全部实验都在 $N \\in [60, 100]$ 上完成。网络科学的实证研究通常在
$10^3$–$10^5$ 节点量级，因此审稿人必然会问：
**这些结论是「机制」还是「小图上的偶然」？**
本实验在 $N \\in \\{60, 100, 250, 500, 1000\\}$ 上重跑旗舰引擎，
检验关键结论是否随规模稳定，并给出趋势的斜率与置信区间。

规模与密度的混淆（复核新增的阻断项）
------------------------------------
引擎默认拓扑是 ER 且 ``initial_density=0.08`` **固定**。于是 $|E| = O(N^2)$、
平均度 $2p(N-1) = O(N)$ —— **规模每翻十倍，密度也翻十倍**。
在这样的配置里，「规模效应」与「密度效应」完全纠缠，任何斜率都无法归因。

因此本实验跑**两条臂**，把两者拆开：

* **密度匹配臂（matched，主结果）**：固定**平均入度** $\\bar{d}$，令
  $p(N) = \\bar{d} / (N-1)$，使各规模的密度**恒定**。
  锚点取论文主实验的 $N=100, p=0.08$，其期望平均入度为
  $\\bar{d} = 0.08 \\times 99 = 7.92$；因此 $N=100$ 处的密度匹配配置与
  论文主实验几乎完全重合，$N=60\\ldots1000$ 是它的密度保持延拓。
  这一臂回答的是：**把同一个机制放到 16 倍大的图上，结论还在吗？**
* **固定 $p$ 臂（fixed-p，混淆探针）**：保持 $p=0.08$ 不变，
  密度随 $N$ 线性上升。这一臂回答的是：
  **如果按「朴素扩规模」的做法（同时放大密度），结论会怎么变？**
  若两臂结论一致，则「结论与规模、密度均无关」；若不一致，则差异必须归因到密度。

此外还包含两个辅助实验：
* **真实 SNAP Higgs 子图**（$N=1000$，10 种子）：真实数据锚点，密度更低（$\\approx 0.015$）；
* **收敛性校验**（$N=250$，$T=30$ vs $T=60$）：确认 60 轮已足够，斜率不是截断假象。

检验的关键量
------------
1. ``Q_end``：模块度是否仍被双极上界 $Q \\le 0.5$ 约束（规模变大后更大图通常更容易
   形成更多社群，若 $Q$ 随 $N$ 单调上升，说明 H1 的证伪理由与规模有关）；
2. ``bridger_gap``：Bridger 与其余节点的留存率差（H2 的核心量）；
3. ``rupture_asymmetry``：非对称断裂率（Phase 4 归因结论是否规模不变）；
4. ``giant_frac``：连通性。

统计口径：每个 $N$ 上跑 ``--seeds`` 个独立种子，报告均值与 95% CI；
趋势用**种子级**配对/回归（每个种子在各个 $N$ 上用同一 seed ⇒ 初始图同源），
并报告斜率的 95% CI 与是否显著异于 0。
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import stats

from .engine import CoEvolutionEngine
from .metrics import engine_metrics, init_incident_edges, retention_by_node
from .expkit import mean_ci

BASE_CFG = dict(base_tolerance=0.4, bridger_tolerance=0.7, p_form=0.1,
                 theta_attract=0.4, heterogeneous=True, block_diff=1.2,
                 rec_friction=1.0)

KEY_METRICS = ("modularity_q", "polarization", "giant_frac", "rupture_asymmetry",
               "num_edges", "bridger_gap")

#: 论文主实验的初始密度（$N=100$ 处）与其期望平均入度 $\\bar{d} = p(N-1)$
MAIN_DENSITY = 0.08
MAIN_ANCHOR_N = 100
TARGET_MEAN_INDEGREE = MAIN_DENSITY * (MAIN_ANCHOR_N - 1)   # = 7.92


def matched_p(n: int, target_mean_indegree: float = TARGET_MEAN_INDEGREE) -> float:
    """密度匹配的 $p$：使 ER 图的期望平均入度恒为 ``target_mean_indegree``。"""
    if n < 2:
        raise ValueError("n must be >= 2")
    return min(1.0, target_mean_indegree / (n - 1))


def run_one(seed: int, nodes: int, rounds: int, init_graph=None, init_beliefs=None,
            initial_density: float = MAIN_DENSITY):
    e = CoEvolutionEngine(num_nodes=nodes, seed=seed, init_graph=init_graph,
                           init_beliefs=init_beliefs,
                           initial_density=initial_density, **BASE_CFG)
    init_edges = set(e.graph.g.edges())
    incident = init_incident_edges(e)
    g = e.graph.g
    n_init_edges = len(init_edges)
    for _ in range(rounds):
        e.run_round()
    m = engine_metrics(e, init_edges)
    ret = retention_by_node(e, incident)
    br = [v for i, v in ret.items() if e.agents[i].archetype == "active_bridger"]
    ot = [v for i, v in ret.items() if e.agents[i].archetype != "active_bridger"]
    m["bridger_retention"] = float(np.mean(br)) if br else float("nan")
    m["other_retention"] = float(np.mean(ot)) if ot else float("nan")
    m["bridger_gap"] = m["bridger_retention"] - m["other_retention"]
    # —— 规模实验必须披露密度，否则「规模」与「密度」无法解耦 ——
    m["n_init_edges"] = float(n_init_edges)
    m["init_mean_indegree"] = float(n_init_edges) / max(nodes, 1)
    m["init_density"] = float(n_init_edges) / max(nodes * (nodes - 1), 1)
    m["end_mean_indegree"] = m["num_edges"] / max(nodes, 1)
    m["n_nodes"] = float(nodes)
    return m


def _aggregate(rows, keys=KEY_METRICS):
    """按指标聚合种子级结果。

    ``rupture_asymmetry`` 在没有互惠二元组断裂时是 NaN（口径定义见
    ``metrics.asymmetry_stats``），NaN 不能进均值——否则整格的均值/CI 会被污染成 NaN。
    这里用有限值子集算均值，并显式记录 ``n_valid`` / ``n_total``。
    """
    agg = {}
    for k in keys:
        arr_all = np.array([r[k] for r in rows], dtype=float)
        arr = arr_all[np.isfinite(arr_all)]
        if arr.size == 0:
            agg[k] = {"mean": float("nan"), "ci95": float("nan"), "n_valid": 0,
                      "n_total": int(arr_all.size), "per_seed": [float(x) for x in arr_all]}
            continue
        mean, half = mean_ci(arr)
        agg[k] = {"mean": float(mean), "ci95": float(half), "n_valid": int(arr.size),
                  "n_total": int(arr_all.size),
                  "per_seed": [float(x) for x in arr_all]}
    return agg


# ——— 并行执行：每个种子一个进程（引擎自带独立 RNG，逐位可复现） ———

def _run_seed_task(task):
    """子进程入口：``task = (seed, nodes, rounds, initial_density)``。"""
    seed, nodes, rounds, density = task
    return run_one(seed, nodes, rounds, initial_density=density)


def _run_higgs_task(task):
    """子进程入口：真实 SNAP Higgs 子图。"""
    seed, nodes, rounds = task
    from .datasets_real import load_higgs_real_subgraph
    g, ideo = load_higgs_real_subgraph(num_nodes=nodes, seed=seed)
    return run_one(seed, nodes, rounds, init_graph=g, init_beliefs=ideo)


def _map_parallel(fn, tasks, procs):
    """按种子并行执行 ``fn``，保持输入顺序；``procs<=1`` 时退化为串行。"""
    if procs <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    from multiprocessing import Pool
    with Pool(processes=min(procs, len(tasks))) as pool:
        return pool.map(fn, tasks)


def _trends(per_size_metric, sizes, seeds, logN=None):
    """种子级 log-N 回归：每个种子跨规模做一次线性回归，再对斜率做种子级汇总。"""
    if logN is None:
        logN = np.log(np.array(sizes, dtype=float))
    trends = {}
    for k in KEY_METRICS:
        slopes, pvals = [], []
        for i in range(seeds):
            ys = np.array([per_size_metric[k][n][i] for n in sizes], dtype=float)
            ok = np.isfinite(ys)
            if ok.sum() >= 3:
                res = stats.linregress(logN[ok], ys[ok])
                slopes.append(float(res.slope))
                pvals.append(float(res.pvalue))
        if not slopes:
            continue
        sl = np.array(slopes)
        mean, half = mean_ci(sl)
        trends[k] = {
            "slope_per_logN_mean": float(mean), "ci95": [float(mean - half), float(mean + half)],
            "n_seeds": int(sl.size),
            "frac_slope_negative": float(np.mean(sl < 0)),
            "slope_significant": bool((mean - half) * (mean + half) > 0),
        }
    return trends


def _hypothesis_robustness(by_size, sizes):
    """逐规模判定三条核心结论是否仍然成立（不因规模而失效）。"""
    out = {}
    for n in sizes:
        a = by_size[str(n)]
        q_mean = a["modularity_q"]["mean"]
        q_ci = a["modularity_q"]["ci95"]
        gap = np.array(a["bridger_gap"]["per_seed"], dtype=float)
        gap = gap[np.isfinite(gap)]
        asym = np.array(a["rupture_asymmetry"]["per_seed"], dtype=float)
        asym = asym[np.isfinite(asym)]
        out[str(n)] = {
            "H1_Q_below_0.5": bool(q_mean + q_ci < 0.5),
            "H1_Q_mean": float(q_mean),
            "H1_Q_ci95": float(q_ci),
            "H2_bridger_gap_positive": bool(gap.size > 0 and gap.mean() > 0),
            "H2_bridger_gap_mean": float(gap.mean()) if gap.size else float("nan"),
            "H2_frac_seeds_positive": float(np.mean(gap > 0)) if gap.size else float("nan"),
            "H3_asymmetry_mean": float(asym.mean()) if asym.size else float("nan"),
            "H3_frac_seeds_above_half": float(np.mean(asym > 0.5)) if asym.size else float("nan"),
        }
    return out


def _checkpoint(out, path):
    tmp = path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_arm(sizes, seeds, rounds, density_mode, out, args, procs=1):
    """跑一条密度臂（matched 或 fixed），返回 (该臂的 by_size, trends, robustness)。"""
    label = "density-matched (固定平均入度)" if density_mode == "matched" else "fixed-p (p=0.08 固定)"
    print(f"\n=== 密度臂：{label} | sizes={sizes} seeds={len(seeds)} rounds={rounds} ===")
    by_size = {}
    per_size_metric = {k: {} for k in KEY_METRICS}
    rows_by_size = {}
    t0 = time.time()
    for n in sizes:
        p = matched_p(n) if density_mode == "matched" else MAIN_DENSITY
        tasks = [(s, n, rounds, p) for s in seeds]
        rows = _map_parallel(_run_seed_task, tasks, procs)
        agg = _aggregate(rows)
        for k in KEY_METRICS:
            agg[k]["per_seed"] = [float(x) for x in agg[k]["per_seed"]]
        # 密度披露
        agg["init_mean_indegree_mean"] = float(np.mean([r["init_mean_indegree"] for r in rows]))
        agg["init_density_mean"] = float(np.mean([r["init_density"] for r in rows]))
        agg["p_used"] = float(p)
        agg["n_seeds"] = len(seeds)
        by_size[str(n)] = agg
        rows_by_size[str(n)] = rows
        for k in KEY_METRICS:
            per_size_metric[k][n] = np.array(agg[k]["per_seed"], dtype=float)
        print(f"  N={n:>5}: p={p:.5f} d̄={agg['init_mean_indegree_mean']:.2f} "
              f"Q={agg['modularity_q']['mean']:.3f}±{agg['modularity_q']['ci95']:.3f}  "
              f"A={agg['rupture_asymmetry']['mean']:.3f}  "
              f"gap={agg['bridger_gap']['mean']:+.3f}  "
              f"giant={agg['giant_frac']['mean']:.3f}  "
              f"[{time.time()-t0:.0f}s]")
    trends = _trends(per_size_metric, sizes, len(seeds))
    robustness = _hypothesis_robustness(by_size, sizes)
    return by_size, trends, robustness, rows_by_size


def main():
    ap = argparse.ArgumentParser(description="Phase 7: network-size sensitivity study")
    ap.add_argument("--sizes", type=str, default="60,100,250,500,1000")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--fixed-p-sizes", type=str, default="60,100,250,500",
                    help="固定 p 对照臂的规模列表（留空则跳过该臂）")
    ap.add_argument("--fixed-p-seeds", type=int, default=10)
    ap.add_argument("--fixed-p-rounds", type=int, default=None,
                    help="固定 p 对照臂的轮数（默认与主臂相同）")
    ap.add_argument("--fixed-p-large-sizes", type=str, default="",
                    help="固定 p 对照臂的大规模补充格（成本高，可单独用较少种子）")
    ap.add_argument("--fixed-p-large-seeds", type=int, default=3,
                    help="固定 p 对照臂大规模补充格的种子数（实际 n 会在结果中披露）")
    ap.add_argument("--procs", type=int, default=8,
                    help="并行进程数（每个进程跑一个种子；引擎自带独立 RNG，结果与串行一致）")
    ap.add_argument("--real-higgs", action="store_true",
                    help="额外在真实 SNAP Higgs 子图上跑最大规模")
    ap.add_argument("--real-higgs-sizes", type=str, default="1000")
    ap.add_argument("--convergence-check", action="store_true",
                    help="在中等规模上做 T/2 vs T 收敛性校验")
    ap.add_argument("--out", type=str, default="results/phase7_scaling.json")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = list(range(args.seeds))
    fixed_sizes = [int(x) for x in args.fixed_p_sizes.split(",") if x.strip()]
    fixed_large_sizes = [int(x) for x in args.fixed_p_large_sizes.split(",") if x.strip()]
    fixed_rounds = args.fixed_p_rounds or args.rounds
    higgs_sizes = [int(x) for x in args.real_higgs_sizes.split(",") if x.strip()]

    out = {
        "args": vars(args),
        "design": {
            "sizes": sizes,
            "seeds": args.seeds,
            "rounds": args.rounds,
            "pairing": "同一 seed 在各规模上共享同源初始图（初始图规模不同，种子同源）",
            "statistical_unit": "seed",
            "density_arms": {
                "matched": {
                    "rule": "固定平均入度 d̄=7.92（= 论文主实验 N=100 @ p=0.08 的期望值），p(N)=d̄/(N-1)",
                    "anchor": "N=100 处与论文主实验几乎完全重合",
                    "purpose": "把规模与密度解耦：检验同一机制在更大图上是否成立",
                },
                "fixed_p": {
                    "rule": "固定 p=0.08（引擎默认值），密度随 N 线性上升",
                    "purpose": "朴素扩规模做法的混淆探针：规模与密度同时变化时会怎样",
                },
            },
        },
        "by_size": {},
        "trends": {},
        "robustness": {},
        "arms": {},
    }

    # —— 主臂：密度匹配 ——
    m_by, m_tr, m_rb, _ = run_arm(sizes, seeds, args.rounds, "matched", out, args,
                                   procs=args.procs)
    out["by_size"] = m_by
    out["trends"] = m_tr
    out["robustness"] = m_rb
    out["arms"]["matched"] = {"by_size": m_by, "trends": m_tr, "robustness": m_rb}
    _checkpoint(out, args.out)

    # —— 对照臂：固定 p ——
    if fixed_sizes or fixed_large_sizes:
        all_fixed_sizes = list(fixed_sizes) + list(fixed_large_sizes)
        # 大规模补充格用较少的种子（成本披露在 n_seeds 字段里）
        rows_by_n = {}
        for n in all_fixed_sizes:
            use_seeds = (list(range(args.fixed_p_seeds)) if n in fixed_sizes
                         else list(range(args.fixed_p_large_seeds)))
            p = MAIN_DENSITY
            tasks = [(s, n, fixed_rounds, p) for s in use_seeds]
            rows = _map_parallel(_run_seed_task, tasks, args.procs)
            rows_by_n[n] = rows
        # 按规模统一汇总（每个规模的种子数可能不同，趋势只用于方向性）
        f_by = {}
        per_size_metric = {k: {} for k in KEY_METRICS}
        for n in all_fixed_sizes:
            rows = rows_by_n[n]
            agg = _aggregate(rows)
            agg["init_mean_indegree_mean"] = float(np.mean([r["init_mean_indegree"] for r in rows]))
            agg["init_density_mean"] = float(np.mean([r["init_density"] for r in rows]))
            agg["p_used"] = float(MAIN_DENSITY)
            agg["n_seeds"] = len(rows)
            f_by[str(n)] = agg
            for k in KEY_METRICS:
                per_size_metric[k][n] = np.array(agg[k]["per_seed"], dtype=float)
            print(f"  [fixed-p] N={n:>5}: d̄={agg['init_mean_indegree_mean']:.2f} "
                  f"Q={agg['modularity_q']['mean']:.3f}±{agg['modularity_q']['ci95']:.3f} "
                  f"(n={len(rows)})")
        # 趋势：只对种子数一致的前缀规模做回归（避免把不同 n 的格子混在一起）
        trend_sizes = [n for n in fixed_sizes if n in f_by]
        f_tr = _trends(per_size_metric, trend_sizes, args.fixed_p_seeds) if trend_sizes else {}
        f_rb = _hypothesis_robustness(f_by, [n for n in all_fixed_sizes])
        out["arms"]["fixed_p"] = {"by_size": f_by, "trends": f_tr, "robustness": f_rb,
                                  "trend_sizes": trend_sizes}
        _checkpoint(out, args.out)

    # —— 真实 SNAP Higgs 子图 ——
    if args.real_higgs:
        print(f"\n=== 真实 SNAP Higgs 子图 | sizes={higgs_sizes} seeds={len(seeds)} ===")
        for n in higgs_sizes:
            tasks = [(s, n, args.rounds) for s in seeds]
            rows = _map_parallel(_run_higgs_task, tasks, args.procs)
            agg = _aggregate(rows)
            agg["init_mean_indegree_mean"] = float(np.mean([r["init_mean_indegree"] for r in rows]))
            agg["init_density_mean"] = float(np.mean([r["init_density"] for r in rows]))
            agg["n_seeds"] = len(rows)
            key = f"higgs_real_n{n}"
            out["by_size"][key] = agg
            out.setdefault("higgs", {})[key] = agg
            print(f"  真实 Higgs N={n}: d̄={agg['init_mean_indegree_mean']:.2f} "
                  f"Q={agg['modularity_q']['mean']:.3f}±{agg['modularity_q']['ci95']:.3f}  "
                  f"A={agg['rupture_asymmetry']['mean']:.3f}  "
                  f"gap={agg['bridger_gap']['mean']:+.3f}  "
                  f"giant={agg['giant_frac']['mean']:.3f}")
        _checkpoint(out, args.out)

    # —— 收敛性校验：T/2 vs T（中等规模） ——
    if args.convergence_check:
        n_c = sizes[len(sizes) // 2]
        print(f"\n=== 收敛性校验：N={n_c}, T={args.rounds} vs T={args.rounds // 2} ===")
        conv = {}
        for T in (args.rounds // 2, args.rounds):
            tasks = [(s, n_c, T, matched_p(n_c)) for s in seeds]
            rows = _map_parallel(_run_seed_task, tasks, args.procs)
            agg = _aggregate(rows)
            conv[str(T)] = agg
            print(f"  T={T}: Q={agg['modularity_q']['mean']:.4f}±{agg['modularity_q']['ci95']:.4f}  "
                  f"A={agg['rupture_asymmetry']['mean']:.4f}  "
                  f"gap={agg['bridger_gap']['mean']:+.4f}")
        a = np.array(conv[str(args.rounds // 2)]["modularity_q"]["per_seed"], dtype=float)
        b = np.array(conv[str(args.rounds)]["modularity_q"]["per_seed"], dtype=float)
        d = b - a
        mean, half = mean_ci(d)
        _, p_val = stats.ttest_rel(b, a)
        out["convergence_check"] = {
            "n": n_c, "rounds_pair": [args.rounds // 2, args.rounds],
            "modularity_q": conv,
            "paired_diff_Q_T_minus_halfT": {"mean": float(mean),
                                            "ci95": [float(mean - half), float(mean + half)],
                                            "p_value_paired_t": float(p_val)},
        }
        _checkpoint(out, args.out)

    # —— 总判定 ——
    verdicts = {}
    for arm, data in out["arms"].items():
        rb = data["robustness"]
        verdicts[arm] = {
            "all_sizes_H1_Q_below_0.5": all(v["H1_Q_below_0.5"] for v in rb.values()),
            "all_sizes_H2_bridger_gap_positive": all(v["H2_bridger_gap_positive"] for v in rb.values()),
            "Q_slope_significant": bool(data["trends"].get("modularity_q", {}).get("slope_significant", False)),
            "Q_slope_per_logN": data["trends"].get("modularity_q", {}).get("slope_per_logN_mean"),
        }
    out["verdict"] = verdicts

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] {args.out}")
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
