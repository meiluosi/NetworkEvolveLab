"""Phase 3：反事实去极化平台政策干预与 H3 假说检验。
政策定义：
  1. Policy 1 (Diversified RecSys): 多样化跨群推荐 (div_rec=0.15)
  2. Policy 2 (Bridger Subsidy): 跨界桥接者失调衰减扶持 (bridger_subsidy=0.50)
  3. Policy 3 (Hub Rate Limiting): 情绪化 Hub 节点触达限流 (hub_limit=0.50)
  4. Combined (组合治理): 三策齐发 (div_rec=0.15, bridger_subsidy=0.50, hub_limit=0.50)

判定准则 (H3):
  - 跨群有效信息到达率提升 >= 35% (相对提升 relative_reach_gain >= +0.35)
  - 模块度 Q 下降 >= 0.20 (delta_q <= -0.20)
"""
import argparse
import json
import numpy as np
from scipy import stats
from .expkit import run_many, mean_ci, fmt

BASE_CONFIG = dict(nodes=100, rounds=60, theta=0.4, bridger_dtheta=0.3, p_form=0.1,
                  block=1.2, rho=1.0, hetero=True, reach=True)

def eval_policy(base_runs, pol_runs, conf=0.95):
    """计算相对于 Baseline 的 delta_q 与 relative_reach_gain，并给出 95% 置信区间与假设判定"""
    q_base = np.array([r["q_end"] for r in base_runs])
    q_pol = np.array([r["q_end"] for r in pol_runs])
    r_base = np.array([r["reach"] for r in base_runs])
    r_pol = np.array([r["reach"] for r in pol_runs])
    
    delta_q = q_pol - q_base
    # 成对相对到达率增益
    reach_gain = (r_pol - r_base) / np.maximum(r_base, 1e-6)
    
    mq_base, hq_base = mean_ci(q_base, conf)
    mq_pol, hq_pol = mean_ci(q_pol, conf)
    md_q, hd_q = mean_ci(delta_q, conf)
    
    mr_base, hr_base = mean_ci(r_base, conf)
    mr_pol, hr_pol = mean_ci(r_pol, conf)
    mgain, hgain = mean_ci(reach_gain, conf)
    
    # 统计显著性检验 (Paired t-test / Wilcoxon)
    p_q = stats.ttest_rel(q_pol, q_base, alternative="less").pvalue
    p_reach = stats.ttest_rel(r_pol, r_base, alternative="greater").pvalue
    
    # H3 判据：delta_q <= -0.20 且 reach_gain >= 0.35
    h3_supported = bool(md_q <= -0.20 and mgain >= 0.35 and p_q < 0.05 and p_reach < 0.05)
    
    return {
        "q_baseline": f"{mq_base:.3f}±{hq_base:.3f}",
        "q_policy": f"{mq_pol:.3f}±{hq_pol:.3f}",
        "delta_q": f"{md_q:+.3f}±{hd_q:.3f}",
        "delta_q_mean": float(md_q),
        "reach_baseline": f"{mr_base:.3f}±{hr_base:.3f}",
        "reach_policy": f"{mr_pol:.3f}±{hr_pol:.3f}",
        "reach_gain": f"{mgain:+.1%}±{hgain:.1%}",
        "reach_gain_mean": float(mgain),
        "p_value_q": float(p_q),
        "p_value_reach": float(p_reach),
        "h3_supported": h3_supported,
        "polarization": fmt([r["polar"] for r in pol_runs]),
        "cross_edges": fmt([r["cross"] for r in pol_runs]),
        "num_edges": fmt([r["edges"] for r in pol_runs])
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Counterfactual Policy Interventions")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--out", type=str, default="results/phase3.json")
    args = parser.parse_args()
    
    seeds = list(range(args.seeds))
    print(f"=== Phase 3: 反事实去极化干预评估 (Seeds={args.seeds}, Rounds={args.rounds}) ===")
    
    base_cfg = {**BASE_CONFIG, "rounds": args.rounds}
    print("Running Baseline (无平台干预)...", flush=True)
    base_runs = run_many(base_cfg, seeds)
    
    policies = {
        "Policy 1 (去中心化推荐 15%)": {"div_rec": 0.15},
        "Policy 2 (桥接者扶持 50%)": {"bridger_subsidy": 0.50},
        "Policy 3 (Hub限流 50%)": {"hub_limit": 0.50},
        "Combined (三策组合治理)": {"div_rec": 0.15, "bridger_subsidy": 0.50, "hub_limit": 0.50}
    }
    
    results = {"baseline": {
        "q_end": fmt([r["q_end"] for r in base_runs]),
        "reach": fmt([r["reach"] for r in base_runs]),
        "polarization": fmt([r["polar"] for r in base_runs]),
        "cross_edges": fmt([r["cross"] for r in base_runs]),
        "num_edges": fmt([r["edges"] for r in base_runs])
    }, "policies": {}, "dose_response_div_rec": {}, "dose_response_hub_limit": {}}
    
    print("\nPolicy Name | Q_end | Delta Q (p) | Reach | Reach Gain (p) | H3 Passed?")
    print("-" * 80)
    for name, pol_kw in policies.items():
        pol_cfg = {**base_cfg, **pol_kw}
        pol_runs = run_many(pol_cfg, seeds)
        ev = eval_policy(base_runs, pol_runs)
        results["policies"][name] = ev
        print(f"{name:<24} | {ev['q_policy']} | {ev['delta_q']} ({ev['p_value_q']:.3f}) | {ev['reach_policy']} | {ev['reach_gain']} ({ev['p_value_reach']:.3f}) | {ev['h3_supported']}")

    # 剂量响应曲线 1：多样化推荐比例扫描
    print("\n--- 剂量响应分析 1: 多样化推荐比例扫描 (div_rec in [0.05, 0.10, 0.15, 0.20, 0.30]) ---")
    for d_rate in [0.05, 0.10, 0.15, 0.20, 0.30]:
        d_cfg = {**base_cfg, "div_rec": d_rate}
        d_runs = run_many(d_cfg, seeds)
        ev = eval_policy(base_runs, d_runs)
        results["dose_response_div_rec"][str(d_rate)] = ev
        print(f"div_rec={d_rate:<4} | Q={ev['q_policy']} (Δ={ev['delta_q']}) | Reach={ev['reach_policy']} (Gain={ev['reach_gain']})")

    # 剂量响应曲线 2：Hub 限流比例扫描
    print("\n--- 剂量响应分析 2: Hub 限流比例扫描 (hub_limit in [0.20, 0.40, 0.50, 0.60, 0.80]) ---")
    for h_rate in [0.20, 0.40, 0.50, 0.60, 0.80]:
        h_cfg = {**base_cfg, "hub_limit": h_rate}
        h_runs = run_many(h_cfg, seeds)
        ev = eval_policy(base_runs, h_runs)
        results["dose_response_hub_limit"][str(h_rate)] = ev
        print(f"hub_limit={h_rate:<4} | Q={ev['q_policy']} (Δ={ev['delta_q']}) | Reach={ev['reach_policy']} (Gain={ev['reach_gain']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Phase 3 results written to {args.out}")

if __name__ == "__main__":
    main()
