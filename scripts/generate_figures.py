"""生成学术论文出版级矢量与高清位图（Figure 1 - Figure 7）
数据来源：
  - results/phase0_sweep.json & results/robustness_sweep.json -> Fig 1: 相变曲线
  - results/phase1b.json -> Fig 2: H2 假说消融小提琴/柱状图
  - results/phase3.json -> Fig 3: 平台治理反事实干预与剂量响应（含到达率面板）
  - results/phase2_offline.json & results/phase2_compare.json -> Fig 4: 语义认知与非对称撕裂
  - results/sota_benchmark.json -> Fig 5: 多方法在评测数据集上的横向对比（模型/数据集列表动态读取）
  - 5 大公开评测数据集 (+ 真实 SNAP Higgs 公开子图) -> Fig 6: 数据集拓扑与微观特征
  - results/phase4_attribution.json -> Fig 7: 非对称撕裂机制归因消融
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
    if isinstance(val_str, (int, float)):
        return float(val_str), 0.0
    if "±" in val_str:
        m, h = val_str.split("±")
        return float(m), float(h)
    return float(val_str), 0.0

def _theta_star(thetas, values, frac=0.5):
    """临界容忍度：values 跌破 frac * values[0] 的最小 θ（相邻点线性插值）。

    与 phase1c_threshold.py 的判据完全一致，保证图与实验口径一致。
    """
    level = frac * values[0]
    for i, v in enumerate(values):
        if v <= level:
            if i == 0:
                return float(thetas[0])
            v0, v1 = values[i - 1], v
            t0, t1 = thetas[i - 1], thetas[i]
            if v0 == v1:
                return float(t1)
            return float(t0 + (v0 - level) / (v0 - v1) * (t1 - t0))
    return float(thetas[-1])


def plot_fig1():
    """Fig 1: 拓扑相变曲线与理论极限上界

    三面板：
      (A) Phase 0 连续容忍度扫描（Q 与极化），临界阈值由曲线本身推导
      (B) 跨拓扑的终态模块度（细网格，Phase 1c）
      (C) 跨拓扑的巨型连通分量占比 —— 真正的「碎裂—连通」相变，含各拓扑实测 θ*
    """
    p0_file = "results/phase0_sweep.json"
    p1c_file = "results/phase1c_threshold.json"

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
    ax1, ax2, ax3 = axes

    # 子图 1: Phase 0 连续容忍度扫描
    theta_star_p0 = None
    if os.path.exists(p0_file):
        d0 = json.load(open(p0_file))
        rows = d0["rows"]
        thetas = [r["theta"] for r in rows]
        q_ends = [r["modularity_q"] for r in rows]
        polars = [r["polarization"] for r in rows]

        ax1.plot(thetas, q_ends, "o-", color="#1f77b4", linewidth=2, label="Newman Modularity $Q$")
        ax1.plot(thetas, polars, "s--", color="#d62728", linewidth=2, label="Polarization Index $\\Pi$")
        ax1.axhline(0.500, color="gray", linestyle=":", linewidth=1.5,
                    label="Bipolar bound ($K=2$): $Q \\leq 0.50$")

        # 临界阈值：由曲线本身推导（Q 跌破起点一半），不再硬编码
        theta_star_p0 = _theta_star(thetas, q_ends)
        ax1.axvline(theta_star_p0, color="orange", linestyle="-.", alpha=0.85,
                    label=r"Measured $\theta^* = %.3f$ ($Q$ collapse)" % theta_star_p0)
        ax1.set_xlabel(r"Cognitive Tolerance Threshold $\theta$")
        ax1.set_ylabel("Macro Metric Value")
        ax1.set_title("(A) Phase Transition & Order Parameter")
        ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax1.set_ylim(-0.05, 0.60)

    colors = {"erdos_renyi": "#1f77b4", "scale_free": "#2ca02c",
              "small_world": "#9467bd", "empirical_bipolar": "#e377c2"}
    labels = {"erdos_renyi": "Erdos-Renyi", "scale_free": "Scale-Free (BA)",
              "small_world": "Small-World (WS)", "empirical_bipolar": "Empirical Bipolar"}

    if os.path.exists(p1c_file):
        d1c = json.load(open(p1c_file))
        band = d1c.get("preregistered_band", [0.50, 0.55])
        for topo, rec in d1c.get("topologies", {}).items():
            curve = rec["curve"]
            th_list = [float(t) for t in sorted(curve.keys(), key=float)]

            # (B) 终态模块度
            means, errs = [], []
            for th in th_list:
                m, h = parse_mean_h(curve[str(th)]["q_end"])
                means.append(m)
                errs.append(h)
            ax2.errorbar(th_list, means, yerr=errs, fmt="o-", capsize=3, linewidth=2,
                         color=colors.get(topo, "black"), label=labels.get(topo, topo))

            # (C) 巨型连通分量占比 —— 真正的碎裂相变
            gm, gh = [], []
            for th in th_list:
                m, h = parse_mean_h(curve[str(th)]["giant_frac"])
                gm.append(m)
                gh.append(h)
            ax3.errorbar(th_list, gm, yerr=gh, fmt="o-", capsize=3, linewidth=2,
                         color=colors.get(topo, "black"), label=labels.get(topo, topo))
            ts = rec.get("theta_frag", {})
            if ts.get("mean") is not None:
                ax3.axvline(ts["mean"], color=colors.get(topo, "black"), linestyle=":",
                            alpha=0.7, linewidth=1.4)
                ax3.annotate(r"$\theta^*=%.2f$" % ts["mean"],
                             xy=(ts["mean"], 0.06), fontsize=8, color=colors.get(topo, "black"),
                             rotation=90, ha="right", va="bottom")

    ax2.axhline(0.500, color="gray", linestyle=":", linewidth=1.5, label="Bipolar bound ($K=2$): 0.50")
    ax2.set_xlabel(r"Cognitive Tolerance Threshold $\theta$")
    ax2.set_ylabel(r"Final Modularity $Q$")
    ax2.set_title("(B) Cross-Topology Final Modularity")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.set_ylim(0.0, 0.70)

    ax3.axhline(0.99, color="gray", linestyle=":", linewidth=1.5,
                label="Global connectivity (giant $\\geq 0.99$)")
    if os.path.exists(p1c_file):
        band = json.load(open(p1c_file)).get("preregistered_band", [0.50, 0.55])
        ax3.axvspan(band[0], band[1], color="orange", alpha=0.12,
                    label="Previously claimed band $[%.2f, %.2f]$" % (band[0], band[1]))
    ax3.set_xlabel(r"Cognitive Tolerance Threshold $\theta$")
    ax3.set_ylabel("Giant Component Fraction")
    ax3.set_title("(C) Shattering to Connectivity Transition")
    ax3.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax3.set_ylim(-0.02, 1.08)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_phase_transition.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig1_phase_transition.svg")
    plt.close()
    if theta_star_p0 is not None:
        print(f"Saved Fig 1 (phase0 theta* = {theta_star_p0:.3f}).")
    else:
        print("Saved Fig 1.")

def plot_fig2():
    """Fig 2: Bridger 机制消融与 H2 证伪图"""
    p1b_file = "results/phase1b.json"
    if not os.path.exists(p1b_file):
        return
    d = json.load(open(p1b_file))
    abl = d["h2_ablation"]

    names = list(abl.keys())
    short_names = [
        "Base (tol=0.7)", "Base (tol=θ)", "Moderate (struct)",
        "Stress (dyn)", "Mod+Stress (0.3)", "Mod+Stress (0.6)",
        "Mod+Stress (0.9)", "Mod+Stress (0.9, tol=θ)"
    ]
    b_ret = [abl[k]["bridger_mean"] for k in names]
    o_ret = [abl[k]["other_mean"] for k in names]

    x = np.arange(len(short_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.0))
    rects1 = ax.bar(x - width/2, b_ret, width, label="Active Bridger", color="#2ca02c", alpha=0.85)
    rects2 = ax.bar(x + width/2, o_ret, width, label="Other Nodes (Hub + Long-tail)", color="#7f7f7f", alpha=0.85)

    ax.set_ylabel("Edge Retention Rate")
    ax.set_title("H2 Hypothesis Ablation: Edge Retention Across Mechanism Variants")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0.0, 1.05)

    for i in range(len(short_names)):
        gap = abl[names[i]]["relative_gap"]
        ax.text(x[i], max(b_ret[i], o_ret[i]) + 0.03, f"{gap:+.1%}", ha="center", fontsize=8.5, color="#333333")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_bridger_ablation.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig2_bridger_ablation.svg")
    plt.close()
    print("Saved Fig 2.")

def plot_fig3():
    """Fig 3: 平台反事实政策干预、剂量响应与跨群到达率（修正后的稳健指标）"""
    p3_file = "results/phase3.json"
    if not os.path.exists(p3_file):
        return
    d = json.load(open(p3_file))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4.3))

    # 子图 1: 各治理策略对模块度的效果
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
    ax1.set_title("(A) Counterfactual Policy Efficacy on $Q$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pols, rotation=20, ha="right")
    ax1.set_ylim(0.30, 0.50)
    ax1.legend(loc="upper right")

    # 子图 2: 剂量响应单调曲线 (div_rec)
    dose = d["results"]["dose_response_div_rec"]
    rates = sorted(float(k) for k in dose.keys())
    q_doses = [parse_mean_h(dose[str(k)]["q_policy"])[0] for k in rates]
    q_d_err = [parse_mean_h(dose[str(k)]["q_policy"])[1] for k in rates]

    ax2.errorbar(rates, q_doses, yerr=q_d_err, fmt="o-", color="#1f77b4", linewidth=2, capsize=4,
                 label="Modularity $Q$")
    ax2.set_xlabel("Diversified Recommendation Rate $\\rho_{\\mathrm{div}}$")
    ax2.set_ylabel("Final Modularity $Q$")
    ax2.set_title(r"(B) Monotonic Dose-Response ($\rho_{\mathrm{div}}$)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0.32, 0.46)

    # 子图 3: 跨群到达率增益（稳健口径：配对差值 + 相对增益）
    labels3 = ["Policy 1", "Policy 2", "Policy 3", "Combined"]
    keys = ["Policy 1 (去中心化推荐 15%)", "Policy 2 (桥接者扶持 50%)",
            "Policy 3 (Hub限流 50%)", "Combined (三策组合治理)"]
    deltas, derrs, rels = [], [], []
    for k in keys:
        ev = d["results"]["policies"][k]
        m, h = parse_mean_h(ev.get("reach_delta", "0±0"))
        deltas.append(m); derrs.append(h)
        rels.append(ev.get("reach_rel_gain_mean", float("nan")))
    x3 = np.arange(len(labels3))
    ax3.bar(x3, rels, yerr=derrs, capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.85)
    ax3.axhline(0.35, color="red", linestyle="--", alpha=0.7, label="H3 threshold: $+35\\%$")
    ax3.axhline(0.0, color="black", linewidth=0.8)
    for i, (dv, rv) in enumerate(zip(deltas, rels)):
        ax3.text(x3[i], rv + (0.03 if rv >= 0 else -0.06), f"$\\Delta$={dv:+.3f}", ha="center", fontsize=8.5)
    ax3.set_ylabel("Cross-Camp Reach: Relative Gain")
    ax3.set_title("(C) Cross-Camp Reach Effect (paired, robust)")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, rotation=15, ha="right")
    ax3.legend(loc="upper left", fontsize=9)

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

    backends = [b for b in ["degroot", "hk(eps=0.6)", "hk(eps=0.9)", "hk(eps=1.2)", "proxy"] if b in d]
    labels = {"degroot": "DeGroot (Scalar)", "hk(eps=0.6)": "HK ($\\epsilon=0.6$)",
              "hk(eps=0.9)": "HK ($\\epsilon=0.9$)", "hk(eps=1.2)": "HK ($\\epsilon=1.2$)",
              "proxy": "Semantic Proxy"}

    q_means = [d[b]["modularity"][0] for b in backends]
    asym_means = [d[b]["rupture_asymmetry"][0] if not np.isnan(d[b]["rupture_asymmetry"][0]) else 0.0 for b in backends]
    asym_errs = [d[b]["rupture_asymmetry"][1] if not np.isnan(d[b]["rupture_asymmetry"][1]) else 0.0 for b in backends]
    small_comm = [d[b]["small_comm_frac"][0] for b in backends]

    # 加入真实 LLM 3-seed 数据（若存在）
    if os.path.exists(comp_file):
        d_llm = json.load(open(comp_file))
        if "llm" in d_llm and len(d_llm["llm"]["modularity"]) > 0:
            labels.append("Real LLM (3-Seeds)")
            q_means.append(float(np.nanmean(d_llm["llm"]["modularity"])))
            vals = np.array(d_llm["llm"]["rupture_asymmetry"], dtype=float)
            finite = vals[np.isfinite(vals)]
            asym_means.append(float(np.mean(finite)) if len(finite) else 0.0)
            asym_errs.append(float(np.std(finite)) if len(finite) else 0.0)
            small_comm.append(float(np.nanmean(d_llm["llm"]["small_comm_frac"])))

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10.5, 5))
    rects1 = ax.bar(x - width, q_means, width, label="Modularity Q", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x, asym_means, width, yerr=asym_errs, capsize=3,
                    label="Asymmetric Rupture $A_{\\mathrm{rupture}}$", color="#d62728", alpha=0.85)
    rects3 = ax.bar(x + width, small_comm, width, label="Small Community Frac", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Metric Value")
    ax.set_title("RQ1: Micro-Cognition Models vs Topological Rupture Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0.0, 1.1)
    ax.text(0.02, 0.60,
            "Note: $A_{\\mathrm{rupture}}>0$ here is produced by the ASYNCHRONOUS\n"
            "single-speaker schedule; see Fig. 7 for the controlled attribution.",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_cognitive_backends.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig4_cognitive_backends.svg")
    plt.close()
    print("Saved Fig 4.")

def plot_fig5():
    """Fig 5: 多方法在评测数据集上的横向对比（模型与数据集列表动态读取，避免与表格漂移）"""
    bench_file = "results/sota_benchmark.json"
    if not os.path.exists(bench_file):
        return
    with open(bench_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = raw["benchmark"]

    datasets = list(data.keys())
    models = list(data[datasets[0]].keys())
    ds_label = {
        "Twitter_Election_2020": "Twitter 2020",
        "SNAP_Higgs_Twitter": "SNAP Higgs\n(surrogate)",
        "Reddit_ConvoKit": "Reddit",
        "DEBATE_Benchmark": "DEBATE",
        "Birdwatch_Community_Notes": "Birdwatch",
        "SNAP_Higgs_Real": "SNAP Higgs\n(REAL)",
    }
    dataset_labels = [ds_label.get(d, d.replace("_", " ")) for d in datasets]
    m_label = {
        "DeGroot": "DeGroot (1974)",
        "Hegselmann_Krause": "Hegselmann-Krause",
        "Friedkin_Johnsen": "Friedkin-Johnsen",
        "Park_et_al_2023": "Park et al. (2023)",
        "Gu_et_al_2025": "Gu et al. (2025)",
        "Ours_CoEvolution_Simplified": "Ours (simplified)",
        "Ours_CoEvolution_Engine": "Ours (CoEvolution Engine)",
    }
    model_labels = [m_label.get(m, m) for m in models]
    palette = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#ff7f0e", "#8c564b", "#1f77b4"]
    colors = [palette[i % len(palette)] for i in range(len(models))]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4.3))
    x = np.arange(len(datasets))
    width = min(0.82 / max(len(models), 1), 0.15)

    for idx, (m, m_lab, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            m_val, h_val = parse_mean_h(data[ds][m]["modularity_q"])
            means.append(m_val); errs.append(h_val)
        offset = (idx - (len(models) - 1) / 2) * width
        ax1.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_lab, color=col, alpha=0.88)

    ax1.set_ylabel("Newman Modularity $Q$")
    ax1.set_title("(A) Community Segregation (Modularity $Q$)")
    ax1.set_xticks(x); ax1.set_xticklabels(dataset_labels, fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0.0, 0.85)

    for idx, (m, m_lab, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            m_val, h_val = parse_mean_h(data[ds][m]["polarization"])
            means.append(m_val); errs.append(h_val)
        offset = (idx - (len(models) - 1) / 2) * width
        ax2.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_lab, color=col, alpha=0.88)

    ax2.set_ylabel(r"Opinion Polarization $\Pi$")
    ax2.set_title(r"(B) Opinion Polarization (Variance $\Pi$)")
    ax2.set_xticks(x); ax2.set_xticklabels(dataset_labels, fontsize=9)
    ax2.set_ylim(0.0, 0.75)

    for idx, (m, m_lab, col) in enumerate(zip(models, model_labels, colors)):
        means, errs = [], []
        for ds in datasets:
            val_str = data[ds][m]["rupture_asymmetry"]
            if val_str == "N/A":
                m_val, h_val = 0.0, 0.0
            else:
                m_val, h_val = parse_mean_h(val_str)
            means.append(m_val); errs.append(h_val)
        offset = (idx - (len(models) - 1) / 2) * width
        ax3.bar(x + offset, means, width, yerr=errs, capsize=3, label=m_lab, color=col, alpha=0.88)

    ax3.set_ylabel(r"Asymmetric Rupture $A_{\mathrm{rupture}}$")
    ax3.set_title(r"(C) Microscopic Severance Asymmetry")
    ax3.set_xticks(x); ax3.set_xticklabels(dataset_labels, fontsize=9)
    ax3.set_ylim(0.0, 1.15)
    ax3.text(0.02, 0.86, "Static/unanimous-break models: 0 or N/A\nAgentic models: unilateral severance persists",
             transform=ax3.transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_sota_benchmark.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig5_sota_benchmark.svg")
    plt.close()
    print("Saved Fig 5.")

def plot_fig6():
    """Fig 6: 评测基准数据集的拓扑与微观特征"""
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
    ax2.fill_between([0.5, 1.0], 0.5, 1.0, color="#2ca02c", alpha=0.1, label="Bridging Consensus Zone (Cross-Aisle)")
    ax2.scatter([], [], c="#2ca02c", label="Bridging Notes (Consensus)")
    ax2.scatter([], [], c="#d62728", label="Partisan Notes (Echo-Chamber)")
    ax2.set_xlabel("Left/Pro Camp Approval Rate")
    ax2.set_ylabel("Right/Con Camp Approval Rate")
    ax2.set_title("(B) Birdwatch Cross-Ideology Consensus")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    # (C) 度分布对比：代理 Higgs vs Twitter 2020 vs 真实 SNAP Higgs 公开子图
    g_higgs = load_snap_higgs(num_nodes=80, seed=42)
    in_degrees_higgs = [d for n, d in g_higgs.in_degree()]

    g_tw, _ = load_twitter_election(num_nodes=80, seed=42)
    in_degrees_tw = [d for n, d in g_tw.in_degree()]

    ax3.hist(in_degrees_higgs, bins=12, alpha=0.55, color="#ff7f0e", label="SNAP Higgs surrogate", density=True)
    ax3.hist(in_degrees_tw, bins=12, alpha=0.55, color="#1f77b4", label="Twitter 2020 (Bipolar BA)", density=True)

    try:
        from src.network_evolve.datasets_real import load_higgs_real_subgraph
        g_real, _ = load_higgs_real_subgraph(num_nodes=80, seed=0)
        in_degrees_real = [d for n, d in g_real.in_degree()]
        ax3.hist(in_degrees_real, bins=12, alpha=0.55, color="#2ca02c",
                 label="SNAP Higgs (REAL public data)", density=True)
    except Exception as exc:  # 真实数据不可用时不阻断绘图
        print(f"[Fig 6] real SNAP Higgs unavailable, skipped: {exc}")

    ax3.set_xlabel(r"Node In-Degree $k_{\mathrm{in}}$")
    ax3.set_ylabel("Empirical Frequency")
    ax3.set_title("(C) Degree Distribution: Surrogate vs Real")
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_empirical_datasets.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig6_empirical_datasets.svg")
    plt.close()
    print("Saved Fig 6.")

def plot_fig7():
    """Fig 7: 非对称撕裂机制归因消融（Phase 4）"""
    p4_file = "results/phase4_attribution.json"
    if not os.path.exists(p4_file):
        print("[Fig 7] results/phase4_attribution.json not found, skipped.")
        return
    d = json.load(open(p4_file))
    arms = d["arms"]
    order = ["scalar_sync", "isotropic_sync", "multidim_sync", "scalar_async",
             "multidim_async", "multidim_async_orig"]
    order = [a for a in order if a in arms]
    labels = [arms[a]["label"] for a in order]
    means = [arms[a]["rupture_asymmetry_mean"] for a in order]
    errs = [arms[a]["rupture_asymmetry_ci95"] for a in order]
    zeros = [arms[a]["frac_seeds_exactly_zero"] for a in order]
    colors = []
    for a in order:
        if a in ("scalar_sync", "isotropic_sync"):
            colors.append("#2ca02c")
        elif a in ("multidim_sync", "multidim_async"):
            colors.append("#1f77b4")
        else:
            colors.append("#ff7f0e")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.3))

    x = np.arange(len(order))
    ax1.bar(x, means, yerr=errs, capsize=4, color=colors, alpha=0.88)
    for i, (m, z) in enumerate(zip(means, zeros)):
        y_base = m + (errs[i] if errs[i] else 0)
        # 零值柱的标签彼此靠得很近，做交错排布避免重叠
        stagger = 0.022 if (m < 1e-9 and i % 2 == 1) else 0.012
        ax1.text(x[i], y_base + stagger,
                 f"{m:.3f}\n(zero in {z:.0%} seeds)", ha="center", fontsize=8.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel(r"Asymmetric Rupture $A_{\mathrm{rupture}}$")
    ax1.set_title("(A) Attribution Ablation: $A_{\\mathrm{rupture}}$ per Arm")
    ax1.set_ylim(0, 0.42)

    # (B) 决定性对比
    contrasts = d.get("contrasts", {})
    keys = [k for k in ["schedule_effect_scalar", "anisotropy_effect_sync", "schedule_effect_multidim"]
            if k in contrasts]
    cmap = {"schedule_effect_scalar": "#ff7f0e", "anisotropy_effect_sync": "#1f77b4",
            "schedule_effect_multidim": "#9467bd"}
    cnames = {"schedule_effect_scalar": "Schedule effect\n(scalar, async − sync)",
              "anisotropy_effect_sync": "Anisotropy effect\n(sync, aniso − iso)",
              "schedule_effect_multidim": "Schedule effect\n(multi-dim, async − sync)"}
    vals = [contrasts[k]["mean_diff"] for k in keys]
    ps = [contrasts[k]["p_value_wilcoxon"] for k in keys]
    npos = [contrasts[k].get("n_positive", 0) for k in keys]
    x2 = np.arange(len(keys))
    ax2.bar(x2, vals, yerr=[contrasts[k]["mean_diff"] * 0 + 0.02 for k in keys], capsize=4,
            color=[cmap[k] for k in keys], alpha=0.88)
    for i, (v, p, npo, k) in enumerate(zip(vals, ps, npos, keys)):
        ax2.text(x2[i], v + 0.012, f"$\\Delta A$={v:+.3f}\np={p:.3g}\n{npo}/{contrasts[k]['n_pairs']} seeds ↑",
                 ha="center", fontsize=8.6)
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([cnames[k] for k in keys], fontsize=9)
    ax2.set_ylabel(r"$\Delta A_{\mathrm{rupture}}$ (paired)")
    ax2.set_title("(B) Decisive Contrasts (paired by seed)")
    ax2.set_ylim(min(-0.05, min(vals) * 1.6), max(0.30, max(vals) * 1.6))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_attribution_ablation.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig7_attribution_ablation.svg")
    plt.close()
    print("Saved Fig 7.")

def plot_fig8():
    """Fig 8: 真 LLM 认知层对照 (Phase 6) + 网络规模敏感性 (Phase 7)"""
    p6_file = "results/phase6_llm_engine.json"
    p7_file = "results/phase7_scaling.json"
    if not (os.path.exists(p6_file) or os.path.exists(p7_file)):
        print("[Fig 8] neither phase6 nor phase7 result found, skipped.")
        return

    n_panels = 2 if (os.path.exists(p6_file) and os.path.exists(p7_file)) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 4.1))
    if n_panels == 1:
        axes = [axes]

    if os.path.exists(p6_file):
        d6 = json.load(open(p6_file))
        ax = axes[0]
        contrasts = d6.get("contrasts", {})
        # 只放尺度无关的结构指标：边数的变化量级（+1.1 条）与其它指标
        # 差一个数量级，画在一起会把所有结构指标压成 0；边数在表格里给出。
        keys = [k for k in ["modularity_q", "polarization", "giant_frac",
                            "rupture_asymmetry"] if k in contrasts]
        labels = {"modularity_q": r"$Q$", "polarization": r"$\Pi$",
                  "giant_frac": "Giant", "rupture_asymmetry": r"$A_{\mathrm{rupture}}$"}
        vals = [contrasts[k]["mean_diff_llm_minus_rule"] for k in keys]
        ps = [contrasts[k]["p_value_paired_t"] for k in keys]
        npos = [contrasts[k].get("n_positive", 0) for k in keys]
        npair = [contrasts[k].get("n_pairs", 0) for k in keys]
        x = np.arange(len(keys))
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.88, width=0.6)
        ax.axhline(0.0, color="black", linewidth=0.9)
        span = max(abs(min(vals)), abs(max(vals)), 1e-3)
        # 标注必须落在 ylim **之内**：画到坐标区外会压住 x 轴刻度标签
        # （此前 Π 与 Q 的标注就掉到了坐标区外，把 $Q$ 刻度字盖住了）。
        off = span * 0.06
        for i, (v, p, npo, npa) in enumerate(zip(vals, ps, npos, npair)):
            txt = f"{v:+.4f}  (p={p:.3g})" if p == p else f"{v:+.4f}"
            ax.text(x[i], v + off if v >= 0 else v - off, txt,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8.4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in keys], fontsize=10)
        ax.set_ylabel(r"$\Delta$ (LLM $-$ rule), paired by seed")
        cov = d6.get("llm_coverage", {})
        ax.set_title("(A) Real LLM vs Rule Backend in the Flagship Engine\n"
                     f"coverage {cov.get('coverage_of_interactions', 0) * 100:.2f}%, "
                     f"parse {cov.get('parse_success_rate', 0) * 100:.0f}%", fontsize=10.5)
        ax.set_ylim(min(vals) - span * 0.55, max(vals) + span * 0.42)

    if os.path.exists(p7_file):
        d7 = json.load(open(p7_file))
        by_size = d7.get("by_size", {})
        trends = d7.get("trends", {})
        arms7 = d7.get("arms", {})
        m7 = (arms7.get("matched", {}) or {}).get("by_size", {})
        f7 = (arms7.get("fixed_p", {}) or {}).get("by_size", {})
        ax = axes[-1]
        sizes = sorted((int(k) for k in by_size.keys() if not k.startswith("higgs")))

        def _series(store):
            ns = sorted((int(k) for k in store if not str(k).startswith("higgs")))
            if not ns:
                return None
            qs = [store[str(n)]["modularity_q"]["mean"] for n in ns]
            err = [store[str(n)]["modularity_q"]["ci95"] for n in ns]
            deg = [store[str(n)].get("init_mean_indegree_mean", float("nan")) for n in ns]
            return ns, qs, err, deg

        m_ser = _series(m7) or _series(by_size)
        f_ser = _series(f7)
        if m_ser:
            ns, qs, qerr, deg = m_ser
            ax.errorbar(np.log(ns), qs, yerr=qerr, marker="o", capsize=4,
                        color="#1f77b4", linewidth=2.2,
                        label=r"$Q$ --- density-matched ($\bar{d}$ fixed)")
            t = trends.get("modularity_q", {})
            if t:
                xs = np.linspace(np.log(min(ns)), np.log(max(ns)), 50)
                ax.plot(xs, t.get("slope_per_logN_mean", 0) * (xs - xs.mean()) + np.mean(qs),
                        "--", color="#1f77b4", linewidth=1.4, alpha=0.65,
                        label=(r"seed-level $\log N$ fit: "
                               f"slope {t.get('slope_per_logN_mean', 0):+.4f}"))
        if f_ser:
            ns, qs, qerr, deg = f_ser
            ax.errorbar(np.log(ns), qs, yerr=qerr, marker="s", capsize=4, linestyle="--",
                        color="#d62728", linewidth=2.2,
                        label=r"$Q$ --- naive fixed-$p$ (density grows)")
        hk = [k for k in by_size.keys() if k.startswith("higgs")]
        if hk:
            hq = by_size[hk[0]]["modularity_q"]["mean"]
            ax.axhline(hq, color="#2ca02c", linestyle=":", linewidth=2.0,
                       label=(f"real SNAP Higgs ($N$="
                              f"{hk[0].replace('higgs_real_n', '')}): $Q$={hq:.3f}"))
        # 右轴：平均入度 —— 混淆的来源
        ax2 = ax.twinx()
        if m_ser:
            ax2.plot(np.log(m_ser[0]), m_ser[3], ":", color="#1f77b4", linewidth=1.6,
                     alpha=0.8, label="mean in-degree (matched)")
        if f_ser:
            ax2.plot(np.log(f_ser[0]), f_ser[3], ":", color="#d62728", linewidth=1.6,
                     alpha=0.8, label="mean in-degree (fixed-$p$)")
        ax2.set_ylabel("mean in-degree $\\bar{d}$", fontsize=9.5)
        ax2.tick_params(labelsize=8.5)
        ax.set_xlabel(r"$\log N$")
        ax.set_ylabel("Final modularity $Q$")
        ax.set_title("(B) Size scaling: density-matched vs.\ naive fixed-$p$\n"
                     "(the collapse in the red curve is a density artefact)",
                     fontsize=10.0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        # 图例放在中左的空白区：左下角被「平均入度」两条点线占据，
        # 放在 lower left 会压住它们（mean in-degree (matched) 那条）。
        ax.legend(h1 + h2, l1 + l2, fontsize=7.6, loc="center left",
                  bbox_to_anchor=(0.015, 0.36), framealpha=0.94,
                  borderpad=0.4, labelspacing=0.35)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_llm_arm_and_scaling.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/fig8_llm_arm_and_scaling.svg")
    plt.close()
    print("Saved Fig 8.")


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    print(f"All figures generated successfully into {OUT_DIR}/")

    # 同步到论文目录，避免 paper/figures 与 results/figures 两处分叉
    import shutil
    paper_figs = os.path.join("paper", "figures")
    os.makedirs(paper_figs, exist_ok=True)
    synced = 0
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith((".png", ".svg")):
            shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(paper_figs, fname))
            synced += 1
    print(f"Synced {synced} figure files into {paper_figs}/")
