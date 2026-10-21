"""生成学术论文出版级矢量与高清位图（Figure 1 - Figure 6）
数据来源：
  - results/phase0_sweep.json & results/robustness_sweep.json -> Fig 1: 相变曲线
  - results/phase1b.json -> Fig 2: H2 假说消融小提琴/柱状图
  - results/phase3.json -> Fig 3: 平台治理反事实干预与剂量响应
  - results/phase2_offline.json & results/phase2_compare.json -> Fig 4: 语义认知与非对称撕裂
  - results/sota_benchmark.json -> Fig 5: 多方法在真实世界评测数据集上的横向对比
  - 5 大公开评测数据集 (DEBATE, Higgs, Twitter, Reddit, Birdwatch) -> Fig 6: 数据集拓扑与微观特征
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

# 论文出版风格设置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def parse_mean_h(val_str):
    if "±" in val_str:
        m, h = val_str.split("±")
        return float(m), float(h)
    return float(val_str), 0.0

def plot_fig1():
    """Fig 1: 拓扑相变曲线与理论极限上界"""
    p0_file = "results/phase0_sweep.json"
    rob_file = "results/robustness_sweep.json"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 子图 1: Phase 0 连续容忍度扫描
    if os.path.exists(p0_file):
        d0 = json.load(open(p0_file))
        thetas = [r["theta"] for r in d0["rows"]]
        q_ends = [r["modularity_q"] for r in d0["rows"]]
        polars = [r["polarization"] for r in d0["rows"]]
        
        ax1.plot(thetas, q_ends, "o-", color="#1f77b4", linewidth=2, label="Newman Modularity $Q$")
        ax1.plot(thetas, polars, "s--", color="#d62728", linewidth=2, label="Polarization Index $\Pi$")
        ax1.axhline(0.500, color="gray", linestyle=":", linewidth=1.5, label="Bipolar Bound $Q \leq 0.50$")
        ax1.axvline(0.52, color="orange", linestyle="-.", alpha=0.8, label=r"Critical Threshold $\theta^* \approx 0.52$")
        ax1.set_xlabel(r"Cognitive Tolerance Threshold $\theta$")
        ax1.set_ylabel("Macro Metric Value")
        ax1.set_title("(A) Phase Transition & Order Parameter")
        ax1.legend(loc="upper right", frameon=True)
        ax1.set_ylim(-0.05, 0.60)

    # 子图 2: 跨拓扑稳健性扫描
    if os.path.exists(rob_file):
        d_rob = json.load(open(rob_file))
        th_list = [0.35, 0.45, 0.55, 0.65]
        colors = {"erdos_renyi": "#1f77b4", "scale_free": "#2ca02c", 
                  "small_world": "#9467bd", "empirical_bipolar": "#e377c2"}
        labels = {"erdos_renyi": "Erdos-Renyi", "scale_free": "Scale-Free (BA)", 
                  "small_world": "Small-World (WS)", "empirical_bipolar": "Empirical Bipolar"}
        
        for topo, data in d_rob["theta_sweep_by_topology"].items():
            means, errs = [], []
            for th in th_list:
                m, h = parse_mean_h(data[str(th)]["q_end"])
                means.append(m)
                errs.append(h)
            ax2.errorbar(th_list, means, yerr=errs, fmt="o-", capsize=4, linewidth=2, 
                         color=colors.get(topo, "black"), label=labels.get(topo, topo))
        
        ax2.axhline(0.500, color="gray", linestyle=":", linewidth=1.5, label="Bipolar Limit (0.50)")
        ax2.set_xlabel(r"Cognitive Tolerance Threshold $\theta$")
        ax2.set_ylabel(r"Final Modularity $Q$")
        ax2.set_title("(B) Cross-Topology Robustness")
        ax2.legend(loc="upper right", frameon=True)
        ax2.set_ylim(0.15, 0.65)
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_phase_transition.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig1_phase_transition.svg")
    plt.close()
    print("Saved Fig 1.")

def plot_fig2():
    """Fig 2: Bridger 机制消融与 H2 证伪图"""
    p1b_file = "results/phase1b.json"
    if not os.path.exists(p1b_file):
        return
    d = json.load(open(p1b_file))
    abl = d["h2_ablation"]
    
    names = list(abl.keys())
    # 简写展示
    short_names = [
        "Base (tol=0.7)", "Base (tol=θ)", "Moderate (struct)", 
        "Stress (dyn)", "Mod+Stress (0.3)", "Mod+Stress (0.6)", 
        "Mod+Stress (0.9)", "Mod+Stress (0.9, tol=θ)"
    ]
    b_ret = [abl[k]["bridger_mean"] for k in names]
    o_ret = [abl[k]["other_mean"] for k in names]
    
    x = np.arange(len(short_names))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(11, 5))
    rects1 = ax.bar(x - width/2, b_ret, width, label="Active Bridger", color="#2ca02c", alpha=0.85)
    rects2 = ax.bar(x + width/2, o_ret, width, label="Other Nodes (Hub + Long-tail)", color="#7f7f7f", alpha=0.85)
    
    ax.set_ylabel("Edge Retention Rate")
    ax.set_title("H2 Hypothesis Ablation: Edge Retention Across Mechanism Variants")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0.0, 1.05)
    
    # 标出相对差距标注
    for i in range(len(short_names)):
        gap = abl[names[i]]["relative_gap"]
        ax.text(x[i], max(b_ret[i], o_ret[i]) + 0.03, f"{gap:+.1%}", ha="center", fontsize=8.5, color="#333333")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_bridger_ablation.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig2_bridger_ablation.svg")
    plt.close()
    print("Saved Fig 2.")

def plot_fig3():
    """Fig 3: 平台反事实政策干预与剂量响应曲线"""
    p3_file = "results/phase3.json"
    if not os.path.exists(p3_file):
        return
    d = json.load(open(p3_file))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 子图 1: 各治理策略对比
    pols = ["Baseline", "Policy 1 (Rec 15%)", "Policy 2 (Bridger 50%)", "Policy 3 (Hub 50%)", "Combined"]
    q_vals, q_errs = [], []
    
    b_q, b_h = parse_mean_h(d["results"]["baseline"]["q_end"])
    q_vals.append(b_q); q_errs.append(b_h)
    
    for k in ["Policy 1 (去中心化推荐 15%)", "Policy 2 (桥接者扶持 50%)", "Policy 3 (Hub限流 50%)", "Combined (三策组合治理)"]:
        mq, hq = parse_mean_h(d["results"]["policies"][k]["q_policy"])
        q_vals.append(mq); q_errs.append(hq)
        
    x = np.arange(len(pols))
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax1.bar(x, q_vals, yerr=q_errs, capsize=4, color=colors, alpha=0.85)
    ax1.axhline(b_q, color="black", linestyle="--", alpha=0.5, label="Baseline Q")
    ax1.set_ylabel("Final Modularity $Q$")
    ax1.set_title("(A) Counterfactual Policy Efficacy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pols, rotation=20, ha="right")
    ax1.set_ylim(0.30, 0.50)
    ax1.legend(loc="upper right")
    
    # 子图 2: 剂量响应单调曲线 (div_rec)
    dose = d["results"]["dose_response_div_rec"]
    rates = [float(k) for k in dose.keys()]
    q_doses = [parse_mean_h(dose[k]["q_policy"])[0] for k in dose.keys()]
    q_d_err = [parse_mean_h(dose[k]["q_policy"])[1] for k in dose.keys()]
    
    ax2.errorbar(rates, q_doses, yerr=q_d_err, fmt="o-", color="#1f77b4", linewidth=2, capsize=4, label="Modularity $Q$")
    ax2.set_xlabel("Diversified Recommendation Rate $\\rho_{\\mathrm{div}}$")
    ax2.set_ylabel("Final Modularity $Q$")
    ax2.set_title(r"(B) Monotonic Dose-Response Curve ($\rho_{\mathrm{div}}$)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0.32, 0.46)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_policy_dose_response.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig3_policy_dose_response.svg")
    plt.close()
    print("Saved Fig 3.")

def plot_fig4():
    """Fig 4: 语义认知后端对比与非对称撕裂"""
    p2_file = "results/phase2_offline.json"
    comp_file = "results/phase2_compare.json"
    if not os.path.exists(p2_file):
        return
    d = json.load(open(p2_file))
    
    backends = ["degroot", "hk(eps=0.6)", "hk(eps=0.9)", "proxy"]
    labels = ["DeGroot (Scalar)", "HK ($\epsilon=0.6$)", "HK ($\epsilon=0.9$)", "Semantic Proxy"]
    
    q_means = [d[b]["modularity"][0] for b in backends]
    asym_means = [d[b]["rupture_asymmetry"][0] if not np.isnan(d[b]["rupture_asymmetry"][0]) else 0.0 for b in backends]
    small_comm = [d[b]["small_comm_frac"][0] for b in backends]
    
    # 加入真实 LLM 3-seed 数据（若存在）
    if os.path.exists(comp_file):
        d_llm = json.load(open(comp_file))
        if "llm" in d_llm and len(d_llm["llm"]["modularity"]) > 0:
            labels.append("Real LLM (3-Seeds)")
            q_means.append(float(np.nanmean(d_llm["llm"]["modularity"])))
            asym_means.append(float(np.nanmean(d_llm["llm"]["rupture_asymmetry"])))
            small_comm.append(float(np.nanmean(d_llm["llm"]["small_comm_frac"])))
            
    x = np.arange(len(labels))
    width = 0.28
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width, q_means, width, label="Modularity Q", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x, asym_means, width, label="Asymmetric Rupture $A_{\mathrm{rupture}}$", color="#d62728", alpha=0.85)
    rects3 = ax.bar(x + width, small_comm, width, label="Small Community Frac", color="#2ca02c", alpha=0.85)
    
    ax.set_ylabel("Metric Value")
    ax.set_title("RQ1: Micro-Cognition Models vs Topological Rupture Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0.0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_cognitive_backends.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig4_cognitive_backends.svg")
    plt.close()
    print("Saved Fig 4.")

def plot_fig5():
    """Fig 5: 多方法在真实世界评测数据集上的横向对比 (SOTA Benchmark Comparison)"""
    bench_file = "results/sota_benchmark.json"
    if not os.path.exists(bench_file):
        return
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)["benchmark"]

    datasets = ["Twitter_Election_2020", "SNAP_Higgs_Twitter", "Reddit_ConvoKit"]
    dataset_labels = ["Twitter 2020 Election", "SNAP Higgs Twitter", "Reddit ConvoKit"]
    models = ["DeGroot", "Hegselmann_Krause", "Friedkin_Johnsen", "Park_et_al_2023", "Gu_et_al_2025", "Ours_CoEvolution"]
    model_labels = ["DeGroot (1974)", "Hegselmann-Krause", "Friedkin-Johnsen", "Park et al. (2023)", "Gu et al. (2025)", "Ours (CoEvolution)"]
    colors = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#ff7f0e", "#1f77b4"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    x = np.arange(len(datasets))
    width = 0.13

    # (A) Modularity Q
    for idx, (m, m_label, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            val_str = data[ds][m]["modularity_q"]
            m_val, h_val = parse_mean_h(val_str)
            means.append(m_val)
            errs.append(h_val)
        offset = (idx - 2.5) * width
        ax1.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_label, color=col, alpha=0.88)

    ax1.set_ylabel("Newman Modularity $Q$")
    ax1.set_title("(A) Community Segregation (Modularity $Q$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax1.set_ylim(0.0, 0.85)

    # (B) Polarization
    for idx, (m, m_label, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            val_str = data[ds][m]["polarization"]
            m_val, h_val = parse_mean_h(val_str)
            means.append(m_val)
            errs.append(h_val)
        offset = (idx - 2.5) * width
        ax2.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_label, color=col, alpha=0.88)

    ax2.set_ylabel(r"Opinion Polarization $\Pi$")
    ax2.set_title(r"(B) Opinion Polarization (Variance $\Pi$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_labels)
    ax2.set_ylim(0.0, 0.75)

    # (C) Asymmetric Rupture
    for idx, (m, m_label, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            val_str = data[ds][m]["rupture_asymmetry"]
            if val_str == "N/A":
                m_val, h_val = 0.0, 0.0
            else:
                m_val, h_val = parse_mean_h(val_str)
            means.append(m_val)
            errs.append(h_val)
        offset = (idx - 2.5) * width
        ax3.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_label, color=col, alpha=0.88)

    ax3.set_ylabel(r"Asymmetric Rupture $A_{\mathrm{rupture}}$")
    ax3.set_title(r"(C) Microscopic Severance Asymmetry")
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_labels)
    ax3.set_ylim(0.0, 1.15)
    ax3.text(0.02, 0.88, "Classical: 0 or N/A (symmetric/fixed)\nAgentic: Non-trivial unilateral blocks", 
             transform=ax3.transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_sota_benchmark.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig5_sota_benchmark.svg")
    plt.close()
    print("Saved Fig 5.")

def plot_fig6():
    """Fig 6: 公开评测基准数据集的拓扑与微观特征 (Benchmark Datasets Characterization)"""
    from src.network_evolve.datasets import (
        load_debate_benchmark, load_snap_higgs, 
        load_twitter_election, load_birdwatch_community_notes
    )
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # (A) Twitter 2020 Election vs DEBATE: 意识形态/信念分布
    _, twitter_ideo = load_twitter_election(num_nodes=60, seed=42)
    t_vals = list(twitter_ideo.values())
    
    debates = load_debate_benchmark()
    d_means = [d["human_prior_mean"] for d in debates]
    d_stds = [d["human_prior_std"] for d in debates]
    
    ax1.hist(t_vals, bins=15, color="#1f77b4", alpha=0.65, density=True, label="Twitter 2020 (DW-NOMINATE Bipolar)")
    # Overlay Gaussian for DEBATE benchmark topics
    x_grid = np.linspace(-1, 1, 100)
    for i, (dm, ds) in enumerate(zip(d_means[:2], d_stds[:2])):
        y_norm = 1.0 / (ds * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_grid - dm) / ds) ** 2)
        ax1.plot(x_grid, y_norm, linewidth=1.8, linestyle="--", label=f"DEBATE Topic {i+1} Prior")
    ax1.set_xlabel("Latent Ideology / Belief Position")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("(A) Belief Distribution in Benchmarks")
    ax1.legend(loc="upper center", fontsize=8.5, frameon=True)

    # (B) Birdwatch / Community Notes: 跨意识形态桥接共识散点图
    bw = load_birdwatch_community_notes(num_nodes=60, seed=42)
    notes = bw["notes"]
    pro_app = [n["pro_camp_approval"] for n in notes]
    con_app = [n["con_camp_approval"] for n in notes]
    is_bridge = [n["is_bridging_consensus"] for n in notes]
    
    colors_bw = ["#2ca02c" if b else "#d62728" for b in is_bridge]
    ax2.scatter(pro_app, con_app, c=colors_bw, s=70, alpha=0.85, edgecolors="k", linewidths=0.5)
    ax2.axvline(0.50, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(0.50, color="gray", linestyle=":", alpha=0.7)
    # Fill bridging consensus quadrant
    ax2.fill_between([0.5, 1.0], 0.5, 1.0, color="#2ca02c", alpha=0.1, label="Bridging Consensus Zone (Cross-Aisle)")
    ax2.scatter([], [], c="#2ca02c", label="Bridging Notes (Consensus)")
    ax2.scatter([], [], c="#d62728", label="Partisan Notes (Echo-Chamber)")
    ax2.set_xlabel("Left/Pro Camp Approval Rate")
    ax2.set_ylabel("Right/Con Camp Approval Rate")
    ax2.set_title("(B) Birdwatch Cross-Ideology Consensus")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    # (C) SNAP Higgs vs Twitter Election 拓扑度分布对比
    g_higgs = load_snap_higgs(num_nodes=80, seed=42)
    in_degrees_higgs = [d for n, d in g_higgs.in_degree()]
    
    g_tw, _ = load_twitter_election(num_nodes=80, seed=42)
    in_degrees_tw = [d for n, d in g_tw.in_degree()]
    
    ax3.hist(in_degrees_higgs, bins=12, alpha=0.6, color="#ff7f0e", label="SNAP Higgs (Scale-Free)", density=True)
    ax3.hist(in_degrees_tw, bins=12, alpha=0.6, color="#1f77b4", label="Twitter 2020 (Bipolar BA)", density=True)
    ax3.set_xlabel(r"Node In-Degree $k_{\mathrm{in}}$")
    ax3.set_ylabel("Empirical Frequency")
    ax3.set_title("(C) Degree Distribution across Real Networks")
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_empirical_datasets.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig6_empirical_datasets.svg")
    plt.close()
    print("Saved Fig 6.")

if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    print(f"All figures generated successfully into {OUT_DIR}/")
