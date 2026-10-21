"""跨网络拓扑稳健性验证（Cross-Topology Robustness Sweep）：
系统比较 Erdos-Renyi 随机图、Barabasi-Albert 无标度网络、Watts-Strogatz 小世界网络与实证双极图：
1. 拓扑相变临界性：在幂律分布与小世界聚类下，θ* 是否依然处于 0.50~0.55 区间？
2. H2 假说跨拓扑检验：在无标度网络 Hub 和小世界社群下，Bridger 留存率是否依然体现拓扑韧性？
3. 平台反事实干预跨拓扑稳健性：去中心化推荐（div_rec=0.15）在无标度网络下是否依然单调降低模块度？
"""
import argparse
import json
import numpy as np
from .expkit import run_many, fmt, h2_summary, mean_ci

TOPOLOGIES = ["erdos_renyi", "scale_free", "small_world", "empirical_bipolar"]

def main():
    parser = argparse.ArgumentParser(description="Cross-Topology Robustness Benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--out", type=str, default="results/robustness_sweep.json")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    print(f"=== 跨网络拓扑稳健性验证 (Seeds={args.seeds}, Rounds={args.rounds}) ===")
    
    results = {
        "args": vars(args),
        "theta_sweep_by_topology": {},
        "h2_bridger_by_topology": {},
        "intervention_by_topology": {}
    }

    # 1. 容忍度相变扫描 (θ in [0.35, 0.45, 0.55, 0.65]) 跨拓扑对比
    print("\n--- 1. 拓扑相变临界区扫描 ---")
    print("Topology         | θ=0.35 Q_end   | θ=0.45 Q_end   | θ=0.55 Q_end   | θ=0.65 Q_end")
    print("-" * 75)
    for topo in TOPOLOGIES:
        row = {}
        row_str = f"{topo:<16} | "
        for th in [0.35, 0.45, 0.55, 0.65]:
            cfg = dict(topology_type=topo, theta=th, rounds=args.rounds, block=1.2, p_form=0.1)
            runs = run_many(cfg, seeds)
            q_str = fmt([r["q_end"] for r in runs])
            row[str(th)] = {
                "q_end": q_str,
                "polarization": fmt([r["polar"] for r in runs]),
                "giant_frac": fmt([r["giant"] for r in runs]),
                "cross_edges": fmt([r["cross"] for r in runs])
            }
            row_str += f"{q_str:<14} | "
        results["theta_sweep_by_topology"][topo] = row
        print(row_str)

    # 2. H2 Bridger 留存率跨拓扑对比
    print("\n--- 2. H2 假说跨拓扑检验 (θ=0.40, block=1.2) ---")
    print("Topology         | Bridger Ret    | Other Ret      | Gap(rel) [95% CI]      | p-val   | H2 Passed")
    print("-" * 85)
    for topo in TOPOLOGIES:
        cfg = dict(topology_type=topo, theta=0.40, rounds=args.rounds, block=1.2, p_form=0.1)
        runs = run_many(cfg, seeds)
        h2_res = h2_summary(runs)
        results["h2_bridger_by_topology"][topo] = h2_res
        print(f"{topo:<16} | {h2_res['bridger_mean']:.3f}          | {h2_res['other_mean']:.3f}          | "
              f"{h2_res['relative_gap']:+.1%} [{h2_res['gap_ci95'][0]:+.1%}, {h2_res['gap_ci95'][1]:+.1%}] | "
              f"{h2_res['p_value']:.4f} | {h2_res['H2_supported']}")

    # 3. 平台反事实干预有效性跨拓扑对比
    print("\n--- 3. 平台去中心化推荐有效性 (Baseline vs div_rec=0.15) ---")
    print("Topology         | Baseline Q_end | Policy 1 Q_end | ΔQ (95% CI)    | Reach Gain")
    print("-" * 75)
    for topo in TOPOLOGIES:
        b_cfg = dict(topology_type=topo, theta=0.40, rounds=args.rounds, block=1.2, p_form=0.1, reach=True)
        p_cfg = dict(topology_type=topo, theta=0.40, rounds=args.rounds, block=1.2, p_form=0.1, div_rec=0.15, reach=True)
        b_runs = run_many(b_cfg, seeds)
        p_runs = run_many(p_cfg, seeds)
        
        q_b = np.array([r["q_end"] for r in b_runs])
        q_p = np.array([r["q_end"] for r in p_runs])
        r_b = np.array([r["reach"] for r in b_runs])
        r_p = np.array([r["reach"] for r in p_runs])
        
        delta_q = q_p - q_b
        gain_r = (r_p - r_b) / np.maximum(r_b, 1e-6)
        
        results["intervention_by_topology"][topo] = {
            "baseline_q": fmt(q_b),
            "policy_q": fmt(q_p),
            "delta_q": fmt(delta_q),
            "reach_gain": fmt(gain_r)
        }
        print(f"{topo:<16} | {fmt(q_b):<14} | {fmt(q_p):<14} | {fmt(delta_q):<14} | {fmt(gain_r)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Robustness sweep written to {args.out}")

if __name__ == "__main__":
    main()
