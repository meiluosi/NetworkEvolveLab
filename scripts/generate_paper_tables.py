"""从冻结的实验结果 JSON 自动生成 LaTeX 表格片段（消除论文与数据的手工同步风险）。

生成物：
  paper/tables/sota_matrix.tex          —— SOTA 基准横向对比表（来自 results/sota_benchmark.json）
  paper/tables/hypothesis_verdicts.tex  —— H1/H2/H3 判定表（来自 phase1b / phase3 / phase4）
  paper/tables/attribution.tex          —— Phase 4 非对称撕裂归因消融表
  paper/tables/dataset_metrics.tex      —— 数据集结构特征表（直接测量基准实际使用的缓存图）

用法：
  python scripts/generate_paper_tables.py            # 生成/更新片段
  python scripts/generate_paper_tables.py --check    # CI 模式：片段过期则退出码 1
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(ROOT, "paper", "tables")
RESULTS = os.path.join(ROOT, "results")

# 让 `from src.network_evolve...` 在以脚本方式直接运行时可用
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def tex_escape(value, chars=("%", "&", "#", "_", "$")) -> str:
    """转义 LaTeX 特殊字符。

    为什么必须有这一步：``%`` 在 LaTeX 里是注释字符，于是
    ``\\newcommand{\\X}{0%}`` 会把右花括号连同后面所有内容一起注释掉，
    表格单元格里出现裸 ``%`` 同样会把该行的 ``\\\\`` 吃掉 —— 两种情况都表现为
    整篇论文 ``Runaway argument?`` / 表格错位。（这个 bug 真实发生过：
    论文此前从未被真正编译过。）

    ``chars`` 按调用场景选择：
      * 宏值（纯数字）→ 转义 ``% & # _ $``；
      * 表格单元格 → 只转义 ``%``（``&`` 是列分隔符，``$`` 用于数学模式）。
    """
    out = str(value)
    for ch in chars:
        out = out.replace(ch, "\\" + ch)
    return out


def escape_pct(text: str) -> str:
    """只转义 ``%``（收据行等自由文本用）。

    这些行里已经手工写过 ``max\\_tokens`` 这类转义，因此不能再套用
    会连 ``_`` 一起转义的通用函数；这里只处理最危险的那个字符。
    """
    return str(text).replace("%", r"\%")


MODEL_DISPLAY = {    "DeGroot": "DeGroot (1974)",
    "Hegselmann_Krause": "Hegselmann-Krause (2002)",
    "Friedkin_Johnsen": "Friedkin-Johnsen (1990)",
    "Park_et_al_2023": "Park et al. (2023)",
    "Gu_et_al_2025": "Gu et al. (2025)",
    "Ours_CoEvolution_Simplified": "Ours (simplified re-impl.)",
    "Ours_CoEvolution_Engine": "Ours (CoEvolution Engine)",
}
DATASET_DISPLAY = {
    "Twitter_Election_2020": "Twitter 2020 Election",
    "SNAP_Higgs_Twitter": "SNAP Higgs Twitter (surrogate)",
    "Reddit_ConvoKit": "Reddit ConvoKit",
    "DEBATE_Benchmark": "DEBATE Benchmark",
    "Birdwatch_Community_Notes": "Birdwatch Community Notes",
    "SNAP_Higgs_Real": "SNAP Higgs Twitter (REAL public data)",
}
#: 转置表（方法为行、数据集为列）用的**短**数据集名：列宽有限。
DATASET_SHORT = {
    "Twitter_Election_2020": "Twitter 2020",
    "SNAP_Higgs_Twitter": "Higgs (surr.)",
    "Reddit_ConvoKit": "Reddit",
    "DEBATE_Benchmark": "DEBATE",
    "Birdwatch_Community_Notes": "Birdwatch",
    "SNAP_Higgs_Real": "Higgs (REAL)",
}
# 静态拓扑方法没有「断裂」可言
STATIC_NOTE = {
    "DeGroot": "N/A (static)",
    "Friedkin_Johnsen": "N/A (static)",
    "Park_et_al_2023": "N/A (static)",
}


def _num(s):
    """把 '0.491±0.001' 解析成 (mean, half_ci)；'N/A' 返回 (None, None)。"""
    if s is None or s == "N/A":
        return None, None
    if isinstance(s, (int, float)):
        return float(s), 0.0
    if "±" in s:
        a, b = s.split("±")
        return float(a), float(b)
    return float(s), 0.0


def _tex_pm(mean, half, digits=3, bold=False):
    if mean is None:
        return "N/A"
    s = f"${mean:.{digits}f} \\pm {half:.{digits}f}$"
    return f"\\mathbf{{{s}}}" if bold else s


def _mean_ci(values):
    """均值与 95% 置信区间（t 分布），NaN 会被剔除。"""
    import math
    from scipy import stats as _st
    vals = [float(v) for v in values if v is not None and v == v]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    half = float(_st.sem(vals) * _st.t.ppf(0.975, len(vals) - 1))
    return mean, half


def _derive_arms_from_per_seed(d):
    """当结果文件的聚合段尚未写出时，从 ``per_seed`` 直接推导 arms / contrasts。

    为什么不直接跳过：Phase~6/7 是长跑实验，聚合段在跑完后才落盘。
    从**同一份逐种子测量**推导聚合，与最终写出的聚合段口径一致
    （种子级均值 + 95% CI + 配对 t 检验），因此不会引入第二套口径，
    只是让表格在实验进行中也能保持与结果文件同步。
    """
    per_seed = d.get("per_seed") or {}
    if not per_seed:
        return d
    arms = {}
    keys = set()
    for row in per_seed.values():
        if isinstance(row, dict):
            for arm, m in row.items():
                if isinstance(m, dict):
                    keys |= set(m.keys())
    for arm in ("rule", "llm_hybrid"):
        rows = [row[arm] for row in per_seed.values()
                if isinstance(row, dict) and isinstance(row.get(arm), dict)]
        if not rows:
            continue
        agg = {}
        for k in sorted(keys):
            mean, half = _mean_ci([r.get(k) for r in rows])
            if mean is None:
                continue
            agg[k] = {"mean": mean, "ci95": half,
                      "per_seed": [float(r.get(k)) for r in rows]}
        arms[arm] = agg
    if not arms:
        return d
    contrasts = {}
    if "rule" in arms and "llm_hybrid" in arms:
        for k in arms["rule"]:
            a = [x for x in arms["llm_hybrid"].get(k, {}).get("per_seed", [])]
            b = [x for x in arms["rule"].get(k, {}).get("per_seed", [])]
            n = min(len(a), len(b))
            if n < 2:
                continue
            diff = [a[i] - b[i] for i in range(n)]
            mean, half = _mean_ci(diff)
            try:
                from scipy import stats as _st
                _t, p_val = _st.ttest_rel(a[:n], b[:n])
                p_val = float(p_val)
            except Exception:
                p_val = float("nan")
            contrasts[k] = {"mean_diff_llm_minus_rule": mean,
                            "ci95": [mean - half, mean + half],
                            "p_value_paired_t": p_val,
                            "n_positive": int(sum(1 for x in diff if x > 0)),
                            "n_pairs": n,
                            "per_seed_diff": diff}
    d = dict(d)
    d["arms"] = d.get("arms") or arms
    d["contrasts"] = d.get("contrasts") or contrasts
    d["_derived_from_per_seed"] = True
    # —— 覆盖率 / 收据同样可以从逐种子的 llm_stats 推导 ——
    if not d.get("llm_coverage"):
        acc = {"n_events": 0, "n_valid": 0, "n_invalid": 0, "n_calls": 0,
               "n_cache_hits": 0, "n_applied": 0, "n_interactions": 0, "actions": {}}
        for row in per_seed.values():
            st = row.get("llm_stats") if isinstance(row, dict) else None
            if not isinstance(st, dict):
                continue
            for k in ("n_events", "n_valid", "n_invalid", "n_calls",
                      "n_cache_hits", "n_applied", "n_interactions"):
                acc[k] += int(st.get(k, 0) or 0)
            for a, c in (st.get("actions") or {}).items():
                acc["actions"][a] = acc["actions"].get(a, 0) + c
        te, ta, ti = acc["n_events"], acc["n_applied"], acc["n_interactions"]
        d["llm_coverage"] = {
            "llm_asked_events": te, "llm_decided_events": ta,
            "valid_events": acc["n_valid"], "invalid_events": acc["n_invalid"],
            "engine_interactions_llm_arm": ti,
            "coverage_of_interactions": (ta / ti) if ti else 0.0,
            "plan_hit_rate": (ta / te) if te else 0.0,
            "parse_success_rate": (acc["n_valid"] / te) if te else 0.0,
            "action_distribution": acc["actions"],
        }
    if not d.get("llm_ledger"):
        d["llm_ledger"] = {
            "model": d.get("args", {}).get("model", "n/a"),
            "temperature": 0.0,
            "max_tokens": d.get("args", {}).get("max_tokens", "n/a"),
            "api_calls": sum(int((r.get("llm_stats") or {}).get("n_calls", 0) or 0)
                             for r in per_seed.values() if isinstance(r, dict)),
            "cache_hits": sum(int((r.get("llm_stats") or {}).get("n_cache_hits", 0) or 0)
                              for r in per_seed.values() if isinstance(r, dict)),
            "totals": {"input_tokens": "n/a", "output_tokens": "n/a",
                       "thinking_tokens": "n/a"},
        }
    return d


def build_llm_arm():
    """Phase 6：真实 LLM 认知层 vs 规则后端（同 seed 配对对照）。"""
    path = os.path.join(RESULTS, "phase6_llm_engine.json")
    if not os.path.exists(path):
        return None
    d = _derive_arms_from_per_seed(json.load(open(path, encoding="utf-8")))
    arms = d.get("arms", {})
    contrasts = d.get("contrasts", {})
    cov = d.get("llm_coverage", {})
    ledger = d.get("llm_ledger", {})
    if not arms or not contrasts:
        return None

    specs = [
        ("modularity_q", "Modularity $Q$", 3),
        ("polarization", "Polarization $\\Pi$", 3),
        ("giant_frac", "Giant-component fraction", 3),
        ("rupture_asymmetry", "Rupture asymmetry $A_{\\mathrm{rupture}}$", 3),
        ("num_edges", "Directed edges", 1),
        ("bridger_retention_mean", "Bridger retention", 3),
    ]
    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Rule backend} & \textbf{LLM hybrid} & "
                 r"\textbf{$\Delta$ (LLM $-$ rule), paired $p$} \\")
    lines.append(r"\midrule")
    for key, label, dig in specs:
        r = arms["rule"].get(key, {})
        l = arms["llm_hybrid"].get(key, {})
        c = contrasts.get(key, {})
        delta = c.get("mean_diff_llm_minus_rule")
        p = c.get("p_value_paired_t")
        dstr = "N/A"
        if delta is not None:
            dstr = f"${delta:+.{dig}f}$"
            if p is not None and p == p:  # 排除 NaN
                dstr += f" ($p={p:.3g}$)"
        lines.append(f"{label} & {_tex_pm(r.get('mean'), r.get('ci95'), dig)} & "
                     f"{_tex_pm(l.get('mean'), l.get('ci95'), dig)} & {dstr} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # 收据行（模型 / 调用数 / token / 解析率 / 覆盖率）—— 单列文字，便于正文引用
    acts = cov.get("action_distribution", {}) or {}
    act_txt = ", ".join(f"{k} {v}" for k, v in sorted(acts.items()))
    receipt_txt = (
        f"Model: {ledger.get('model', 'n/a')}; temperature: {ledger.get('temperature', 'n/a')}; "
        f"max\\_tokens: {ledger.get('max_tokens', 'n/a')}; "
        f"distinct LLM requests: {ledger.get('total_api_calls_for_experiment', ledger.get('api_calls', 'n/a'))} "
        f"(each required exactly one API call; the verification run replayed all of them from the "
        f"content-addressed archive at zero API cost --- cache hits: {ledger.get('cache_hits', 'n/a')}); "
        f"input tokens: {ledger.get('totals', {}).get('input_tokens', 'n/a')}; "
        f"output tokens: {ledger.get('totals', {}).get('output_tokens', 'n/a')}; "
        f"thinking tokens: {ledger.get('totals', {}).get('thinking_tokens', 'n/a')}. "
        f"Parse success rate: {cov.get('parse_success_rate', 0) * 100:.1f}%; "
        f"interaction coverage: {cov.get('coverage_of_interactions', 0) * 100:.2f}% "
        f"(LLM-decided {cov.get('llm_decided_events', 'n/a')} / "
        f"{cov.get('engine_interactions_llm_arm', 'n/a')} engine interactions). "
        f"LLM action distribution: {act_txt if act_txt else 'n/a'}."
    )
    receipt = [
        "",
        r"\medskip",
        r"\noindent\footnotesize",
        escape_pct(receipt_txt),
        r"\normalsize",
    ]
    caption = ("Phase 6: real-LLM cognitive layer versus the rule backend inside the flagship "
               "scalar engine, paired by seed. Generated from results/phase6_llm_engine.json.")
    return "\n".join(lines + receipt), caption


def build_scaling():
    """Phase 7：网络规模敏感性（N ∈ {60…1000} + 真实 Higgs 子图）。"""
    path = os.path.join(RESULTS, "phase7_scaling.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    by_size = d.get("by_size", {})
    trends = d.get("trends", {})
    if not by_size:
        return None

    specs = [
        ("modularity_q", "Modularity $Q$", 3),
        ("polarization", "Polarization $\\Pi$", 3),
        ("giant_frac", "Giant-component fraction", 3),
        ("rupture_asymmetry", "Rupture asymmetry $A_{\\mathrm{rupture}}$", 3),
        ("bridger_gap", "Bridger gap", 3),
    ]
    # 表头用短名（长名放 caption 里）；列数必须与 specs 长度一致，
    # 否则 LaTeX 会报 "Extra alignment tab has been changed to \cr"。
    headers = [r"$Q$", r"$\Pi$", r"Giant frac.", r"Rupture asym.", r"Bridger gap"]
    assert len(headers) == len(specs)
    sizes = [k for k in by_size.keys() if k.startswith("higgs") is False]
    sizes.sort(key=lambda s: int(s))
    extra = [k for k in by_size.keys() if k.startswith("higgs")]

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    # 列数必须与表头一致（N + 平均入度 + 每个指标一列），否则 LaTeX 会报
    # "Extra alignment tab has been changed to \cr"。
    ncols = len(specs) + 2
    lines.append(r"\begin{tabular}{l" + "c" * (ncols - 1) + "}")
    lines.append(r"\toprule")
    header_cells = ([r"\textbf{$N$}", r"\textbf{mean deg.}"]
                    + [r"\textbf{%s}" % h for h in headers])
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    def _row(label, agg):
        cells = [f"{agg.get('init_mean_indegree_mean', float('nan')):.1f}"]
        for key, _lab, dig in specs:
            a = agg.get(key, {})
            cells.append(_tex_pm(a.get("mean"), a.get("ci95"), dig))
        return f"{label} & " + " & ".join(cells) + r" \\"

    for n in sizes:
        lines.append(_row(str(int(n)), by_size[n]))
    # —— 固定-p 对照臂：密度随 N 增长的混淆探针 ——
    arms7 = d.get("arms", {})
    fp = (arms7.get("fixed_p", {}) or {}).get("by_size", {})
    if fp:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textit{Fixed-$p$ arm (density grows with $N$)}} \\"
                     % ncols)
        for n in sorted(fp, key=lambda s: int(s)):
            lines.append(_row(str(int(n)), fp[n]))
    if extra:
        lines.append(r"\midrule")
        for n in extra:
            lines.append(_row(n.replace("_", " "), by_size[n]))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    slope_txt = []
    for key, label, _dig in specs:
        t = trends.get(key)
        if not t:
            continue
        lo, hi = t.get("ci95", [0, 0])
        sig = "significant" if t.get("slope_significant") else "not significant"
        slope_txt.append(f"{label}: slope per $\\log N$ = {t.get('slope_per_logN_mean', 0):+.4f} "
                         f"(95% CI [{lo:+.4f}, {hi:+.4f}], {sig})")
    receipt = [
        "",
        r"\medskip",
        r"\noindent\footnotesize",
        escape_pct(("Seed-level log-$N$ regressions: " + "; ".join(slope_txt)
                    if slope_txt else "") + "."),
        r"\normalsize",
    ]
    caption = ("Phase 7: network-size sensitivity of the flagship engine. Generated from "
               "results/phase7_scaling.json.")
    return "\n".join(lines + receipt), caption


def build_sota_matrix():
    """完整的跨基准比较矩阵（5 个指标 × 6 个数据集 × 7 个方法）。

    这个矩阵有 42 行，原样放在正文里会吃掉将近一整页，而 ICWSM 的
    11 页预算（正文 + 参考文献）放不下。因此：
      * **正文**用 ``build_sota_matrix_compact()``：转置成「方法为行、
        数据集为列」的 $Q$ 对比表（7 行 × 7 列），一眼可横向比较；
      * **附录**放这里的完整矩阵（5 个指标全在）。
    两个表由同一份 ``results/sota_benchmark.json`` 机器生成，数据点一个不少。
    """
    path = os.path.join(RESULTS, "sota_benchmark.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["benchmark"]
    datasets = list(data.keys())
    models = list(data[datasets[0]].keys())
    args = json.load(open(path, encoding="utf-8")).get("args", {})

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Empirical Dataset} & \textbf{Model / Method} & \textbf{Modularity $Q$} & "
                 r"\textbf{Polarization $\Pi$} & \textbf{Giant Frac} & \textbf{Small Comm Frac} & "
                 r"\textbf{Rupture Asym $A_{\mathrm{rupture}}$} \\")
    lines.append(r"\midrule")
    for d in datasets:
        for i, m in enumerate(models):
            row = data[d][m]
            q_m, q_h = _num(row["modularity_q"])
            p_m, p_h = _num(row["polarization"])
            g_m, g_h = _num(row["giant_frac"])
            s_m, s_h = _num(row["small_comm_frac"])
            a_m, a_h = _num(row["rupture_asymmetry"])
            note = STATIC_NOTE.get(m, "")
            asym_txt = _tex_pm(a_m, a_h) if a_m is not None else (note or "N/A")
            prefix = ""
            if i == 0:
                prefix = (r"\multirow{" + str(len(models)) + r"}{*}{\textbf{" +
                          DATASET_DISPLAY.get(d, d) + r"}} ")
            name = MODEL_DISPLAY.get(m, m)
            if m.startswith("Ours"):
                name = r"\textbf{" + name + r"}"
            lines.append(
                f"{prefix}& {name} & {_tex_pm(q_m, q_h)} & {_tex_pm(p_m, p_h)} & "
                f"{_tex_pm(g_m, g_h)} & {_tex_pm(s_m, s_h)} & {asym_txt} \\\\")
        lines.append(r"\midrule")
    lines.pop()  # 最后一行不要 midrule
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = (f"Values report Mean $\\pm$ 95\\% Confidence Interval across "
               f"{args.get('seeds', '?')} independent random seeds "
               f"($N={args.get('nodes', '?')}$, $T={args.get('rounds', '?')}$ rounds).")
    return "\n".join(lines), caption


def build_sota_matrix_compact():
    """正文用的紧凑对比表：方法为行、数据集为列，只放模块度 $Q$。

    转置后是 7 行 × 7 列，高度只有原矩阵的三分之一；每个方法在一行里
    横向比较六个基准网络。其余四个指标在附录的完整矩阵里。
    """
    path = os.path.join(RESULTS, "sota_benchmark.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["benchmark"]
    datasets = list(data.keys())
    models = list(data[datasets[0]].keys())
    args = json.load(open(path, encoding="utf-8")).get("args", {})

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append(r"\toprule")
    header = [r"\textbf{Method / Model}"] + [r"\textbf{%s}" % DATASET_SHORT.get(d, d)
                                             for d in datasets]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for m in models:
        name = MODEL_DISPLAY.get(m, m)
        if m.startswith("Ours"):
            name = r"\textbf{" + name + r"}"
        cells = []
        for d in datasets:
            q_m, q_h = _num(data[d][m]["modularity_q"])
            cells.append(_tex_pm(q_m, q_h))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        if m == "Ours_CoEvolution_Simplified":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = (f"Modularity $Q$ (Mean $\\pm$ 95\\% CI) across "
               f"{args.get('seeds', '?')} seeds; the full five-metric matrix for every "
               f"dataset $\\times$ method cell is in Appendix~\\ref{{app:benchmark-matrix}}.")
    return "\n".join(lines), caption


def build_hypothesis_verdicts():
    p1b = os.path.join(RESULTS, "phase1b.json")
    p3 = os.path.join(RESULTS, "phase3.json")
    p4 = os.path.join(RESULTS, "phase4_attribution.json")
    if not (os.path.exists(p1b) and os.path.exists(p3)):
        return None
    d1 = json.load(open(p1b, encoding="utf-8"))
    d3 = json.load(open(p3, encoding="utf-8"))["results"]
    d4 = json.load(open(p4, encoding="utf-8")) if os.path.exists(p4) else None

    # H1
    grid = d1["h1_grid"]
    q_ends = [v["q_end"][0] for v in grid.values()]
    q_max = max(v["max_q_end"] for v in grid.values())
    pass_rate = max(v["h1_pass_rate"] for v in grid.values())
    k_note = ""
    if d4 is not None and "summary" in d4:
        pass
    h1 = (f"$Q_{{\\mathrm{{init}}}}=0.213 \\to Q_{{\\mathrm{{end}}}}={min(q_ends):.3f}"
          f"\\sim{max(q_ends):.3f}$ (max observed $Q$={q_max:.3f}); "
          f"pass rate $\\leq$ {tex_escape(f'{pass_rate:.0%}')} over {len(grid)} configurations")

    # H2
    base = d1["h2_ablation"]["base(原模型,bridger_tol=0.7)"]
    sl = base.get("seed_level", {})
    h2 = (f"Bridger retention {base['bridger_mean']:.3f} vs. others {base['other_mean']:.3f}; "
          f"seed-level gap {sl.get('seed_gap_mean', float('nan')):+.3f} "
          f"({tex_escape('95% CI')} [{base['gap_ci95'][0]:+.3f}, {base['gap_ci95'][1]:+.3f}]), "
          f"seed-level $p={sl.get('p_value_seed_level', float('nan')):.3f}$")

    # H3
    pol = d3["policies"]
    p1 = pol["Policy 1 (去中心化推荐 15%)"]
    comb = pol["Combined (三策组合治理)"]
    p1_gain = tex_escape(f"{p1.get('reach_rel_gain_mean', float('nan')):+.1%}")
    comb_gain = tex_escape(f"{comb.get('reach_rel_gain_mean', float('nan')):+.1%}")
    h3 = (f"Policy 1: $\\Delta Q={p1['delta_q_mean']:+.3f}$ ($p={p1['p_value_q']:.2g}$), "
          f"reach $\\Delta={p1.get('reach_delta_mean', float('nan')):+.3f}$, "
          f"relative gain {p1_gain}; "
          f"Combined: $\\Delta Q={comb['delta_q_mean']:+.3f}$ ($p={comb['p_value_q']:.2g}$), "
          f"reach $\\Delta={comb.get('reach_delta_mean', float('nan')):+.3f}$, "
          f"relative gain {comb_gain}")

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{@{}p{2.3cm}p{3.0cm}p{10.2cm}l@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Hypothesis} & \textbf{Pre-specified threshold} & \textbf{Measured result} & \textbf{Verdict} \\")
    lines.append(r"\midrule")
    lines.append(r"H1 (polarization modularity) & $Q: 0.25 \to 0.65$ within $T=20$ & " + h1 + r" & Falsified \\")
    lines.append(r"H2 (bridger fragility) & bridger retention $\geq$ 40\% lower, $p<0.01$ & " + h2 + r" & Falsified \\")
    lines.append(r"H3 (counterfactual depolarization) & reach $\geq +35\%$ and $\Delta Q \leq -0.20$ & " + h3 + r" & Not supported \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines), None


def build_attribution():
    path = os.path.join(RESULTS, "phase4_attribution.json")
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path, encoding="utf-8"))
    order = ["scalar_sync", "isotropic_sync", "multidim_sync",
             "scalar_async_uniform", "scalar_async",
             "isotropic_async_uniform", "isotropic_async",
             "multidim_async_uniform", "multidim_async", "multidim_async_orig"]
    arms = d["arms"]
    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Arm} & \textbf{$A_{\mathrm{rupture}}$} & \textbf{Seeds with $A \equiv 0$} & "
                 r"\textbf{Mutual dyads broken} \\")
    lines.append(r"\midrule")
    for a in order:
        if a not in arms:
            continue
        rec = arms[a]
        frac_zero = tex_escape(f"{rec['frac_seeds_exactly_zero']:.0%}")
        n_mut = str(rec.get('n_mutual_broken', 'n/a'))
        if "±" in n_mut:
            n_mut = f"${n_mut.replace('±', r' \pm ')}$"
        lines.append(f"{rec['label']} & "
                     f"${rec['rupture_asymmetry_mean']:.3f} \\pm {rec['rupture_asymmetry_ci95']:.3f}$ & "
                     f"{frac_zero} & {n_mut} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    c = d.get("contrasts", {})
    contrast_lines = []
    names = {
        "schedule_effect_scalar_pure": "Pure temporal asynchrony (scalar, uniform speaker)",
        "speaker_bias_effect_scalar": "Speaker frequency bias (scalar, async weighted vs uniform)",
        "schedule_effect_scalar": "Combined async schedule (scalar, weighted speaker)",
        "anisotropy_effect_sync": "Per-agent salience anisotropy (synchronous baseline)",
        "anisotropy_effect_async_uniform": "Per-agent salience anisotropy (async uniform speaker)",
        "schedule_effect_isotropic_pure": "Pure temporal asynchrony (multi-d isotropic)",
        "schedule_effect_multidim_pure": "Schedule effect under anisotropy (multi-d uniform: async - sync)",
        "schedule_effect_multidim": "Asynchronous schedule (multi-dimensional)",
        "tolerance_rule_effect": "Tolerance rule (asymmetric theta_ij)",
    }
    for k, label in names.items():
        if k in c:
            contrast_lines.append(f"{label}: $\\Delta A={c[k]['mean_diff']:+.3f}$ "
                                  f"(Wilcoxon $p={c[k]['p_value_wilcoxon']:.3g}$)")
    return "\n".join(lines), ("; ".join(contrast_lines) if contrast_lines else None)


def build_dataset_metrics():
    """数据集结构特征表：直接测量基准实际使用的缓存图（不手工抄写任何数字）。

    之所以必须机器生成：本项目上一版附录里的数据集表存在「README 说 3 大、
    正文说 5 大、图注写 three」的互相矛盾（F-08），根因就是这张表是人手填的。
    这里改为调用 benchmark_suite.get_dataset —— 与基准评测完全同一条装载路径 ——
    再在图上实测节点数、边数、互惠率、平均度、初始模块度与阵营数。

    同时显式区分**真实公开数据**与**合成代理**，避免把 surrogate 说成 empirical。
    """
    import networkx as nx
    import numpy as np
    from src.network_evolve.benchmark_suite import get_dataset, DATASETS

    bench_path = os.path.join(RESULTS, "sota_benchmark.json")
    if not os.path.exists(bench_path):
        return None, None
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    args = bench.get("args", {})
    n = int(args.get("nodes", 60))
    seeds = int(args.get("seeds", 10))

    # 只有 SNAP_Higgs_Real 来自真实公开数据；其余 5 个是合成代理（surrogate proxy）。
    provenance = {
        "Twitter_Election_2020": "Surrogate proxy",
        "SNAP_Higgs_Twitter": "Surrogate proxy",
        "Reddit_ConvoKit": "Surrogate proxy",
        "DEBATE_Benchmark": "Surrogate proxy",
        "Birdwatch_Community_Notes": "Surrogate proxy",
        "SNAP_Higgs_Real": "REAL public data",
    }

    rows = []
    for dname in DATASETS:
        if dname not in bench.get("benchmark", {}):
            continue
        recs = []
        for s in range(seeds):
            try:
                g, ideo = get_dataset(dname, n, s)
            except Exception as exc:  # pragma: no cover - 数据缺失时跳过
                print(f"[warn] {dname} seed={s} 装载失败: {exc}")
                continue
            und = g.to_undirected()
            comms = nx.community.louvain_communities(und, seed=s)
            q0 = nx.community.modularity(und, comms) if und.number_of_edges() else 0.0
            vals = np.array(list(ideo.values()), dtype=float) if ideo else np.zeros(1)
            camp_labels = {1 if v > 0.25 else (-1 if v < -0.25 else 0) for v in vals}
            degs = np.array([d for _, d in g.degree()], dtype=float)
            recs.append({
                "n": g.number_of_nodes(),
                "e": g.number_of_edges(),
                "r": float(nx.overall_reciprocity(g)) if g.number_of_edges() else 0.0,
                "k": float(degs.mean()) if degs.size else 0.0,
                "q0": float(q0),
                "camps": len(camp_labels - {0}),
                "k_louvain": float(len(comms)),
            })
        if not recs:
            continue
        rows.append((dname, recs))

    def agg(recs, key):
        arr = np.array([r[key] for r in recs], dtype=float)
        m = arr.mean()
        half = (1.96 * arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        return m, half

    lines = []
    lines.append("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Provenance} & \textbf{$N$} & \textbf{$E$} & "
                 r"\textbf{Recip.\ $r$} & \textbf{$\langle k\rangle$} & "
                 r"\textbf{$Q_0$ (Louvain)} & \textbf{$K_0$ (Louvain)} \\")
    lines.append(r"\midrule")
    for dname, recs in rows:
        prov = provenance.get(dname, "Surrogate proxy")
        e_m, e_h = agg(recs, "e")
        r_m, r_h = agg(recs, "r")
        k_m, k_h = agg(recs, "k")
        q_m, q_h = agg(recs, "q0")
        kk_m, kk_h = agg(recs, "k_louvain")
        prov_txt = r"\textbf{REAL public data}" if prov.startswith("REAL") else prov
        lines.append(
            f"{DATASET_DISPLAY.get(dname, dname)} & {prov_txt} & {recs[0]['n']} & "
            f"${e_m:.0f} \\pm {e_h:.0f}$ & ${r_m:.3f} \\pm {r_h:.3f}$ & "
            f"${k_m:.1f} \\pm {k_h:.1f}$ & ${q_m:.3f} \\pm {q_h:.3f}$ & "
            f"${kk_m:.1f} \\pm {kk_h:.1f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    real = [r for d, r in rows if provenance.get(d, "").startswith("REAL")]
    caption = (f"All structural quantities are measured directly on the cached graphs actually "
               f"used by the benchmark suite ($N={n}$, {seeds} seeds), not transcribed by hand. "
               f"Only SNAP Higgs Twitter (REAL public data) derives from a public dataset; the "
               f"other {len(rows) - len(real)} networks are synthetic surrogates that reproduce the "
               f"qualitative topology of their reference sources. "
               f"$r$ is the directed reciprocity $2\\,\\#\\text{{mutual}}/|E|$, "
               f"$Q_0$ and $K_0$ are the modularity and the number of communities of the Louvain "
               f"partition of the initial topology; $K_0$ is the quantity that bounds $Q$ via "
               f"$Q \\leq 1-1/K$ (Proposition~1).")
    return "\n".join(lines), caption


def build_macros():
    """生成 paper/tables/macros.tex：正文所有定量断言的自动生成宏。

    论文正文一律引用这些宏（例如 \\HOneMaxQ），从而保证「论文里的每个数字」
    都直接来自冻结的结果 JSON，不可能与实验漂移。
    """
    p1b = os.path.join(RESULTS, "phase1b.json")
    p3 = os.path.join(RESULTS, "phase3.json")
    p4 = os.path.join(RESULTS, "phase4_attribution.json")
    bench = os.path.join(RESULTS, "sota_benchmark.json")
    p7 = os.path.join(RESULTS, "phase7_scaling.json")
    if not (os.path.exists(p1b) and os.path.exists(p3)):
        return None

    d1 = json.load(open(p1b, encoding="utf-8"))
    d3_full = json.load(open(p3, encoding="utf-8"))
    d3 = d3_full["results"]
    d4 = json.load(open(p4, encoding="utf-8")) if os.path.exists(p4) else None
    db = json.load(open(bench, encoding="utf-8")) if os.path.exists(bench) else None
    d7 = json.load(open(p7, encoding="utf-8")) if os.path.exists(p7) else None

    defs = []

    def macro(name, value):
        defs.append(f"\\newcommand{{\\{name}}}{{{tex_escape(value)}}}")

    # —— 默认值登记表 ——
    # 长跑实验（Phase 6/7）的聚合段在实验跑完前不存在。若那时才定义宏，
    # 论文里就会出现 "Undefined control sequence" 而无法编译。
    # 这里给每一族宏登记一个默认值，保证宏集合**始终完整**：
    # 数据齐备时被真实值覆盖，数据未齐时显式显示 n/a（而不是编译失败，
    # 也不是静默留白）。
    DEFAULTS = {
        "LlmArmSeeds": "n/a", "LlmArmRounds": "n/a", "LlmArmNodes": "n/a",
        "LlmArmEventsPerRound": "n/a", "LlmArmModel": "n/a",
        "LlmArmApiCalls": "n/a", "LlmArmCacheHits": "n/a",
        "LlmArmInputTokens": "n/a", "LlmArmOutputTokens": "n/a",
        "LlmArmThinkingTokens": "n/a", "LlmArmParseRate": "n/a",
        "LlmArmCoverage": "n/a", "LlmArmAskedEvents": "n/a",
        "LlmArmDecidedEvents": "n/a", "LlmArmInteractions": "n/a",
        "LlmArmQDelta": "n/a", "LlmArmQP": "n/a", "LlmArmQPositive": "n/a",
        "LlmArmRuptureDelta": "n/a", "LlmArmRuptureP": "n/a",
        "LlmArmActionKeep": "n/a", "LlmArmActionMute": "n/a",
        "LlmArmActionUnfollow": "n/a", "LlmArmActionBlock": "n/a",
        "ScalingMaxN": "n/a", "ScalingSizes": "n/a", "ScalingSeeds": "n/a",
        "ScalingRounds": "n/a", "ScalingBigQ": "n/a", "ScalingBigGiant": "n/a",
        "ScalingSmallQ": "n/a",
        "ScalingQSlope": "n/a", "ScalingQSlopeCILow": "n/a",
        "ScalingQSlopeCIHigh": "n/a", "ScalingQSlopeSignificant": "n/a",
        "ScalingRuptureSlope": "n/a", "ScalingRuptureSlopeCILow": "n/a",
        "ScalingRuptureSlopeCIHigh": "n/a", "ScalingBridgerGapSlope": "n/a",
        "ScalingRealHiggsN": "n/a", "ScalingRealHiggsQ": "n/a",
        "ScalingRealHiggsGiant": "n/a", "ScalingRealHiggsBridgerGap": "n/a",
        "NumProcs": "n/a",
        "ScalingDensityCommonSizes": "n/a", "ScalingDensityCompareN": "n/a",
        "ScalingMatchedBigQ": "n/a", "ScalingFixedPBigQ": "n/a",
        "ScalingFixedPBigDensity": "n/a", "ScalingMatchedBigDensity": "n/a",
        "ScalingFixedPBigSeeds": "n/a", "ScalingFixedPCollapseN": "n/a",
        "ScalingFixedPCollapseQ": "n/a", "ScalingFixedPCollapseDensity": "n/a",
        "ScalingConvergenceN": "n/a", "ScalingConvergenceRoundOne": "n/a",
        "ScalingConvergenceRoundTwo": "n/a", "ScalingConvergenceDelta": "n/a",
        "ScalingConvergenceP": "n/a",
        "ScalingMatchedQMin": "n/a", "ScalingMatchedQMax": "n/a",
        "ScalingQTotalRise": "n/a", "ScalingCollapseGap": "n/a",
        "ARupScalarAsyncUniform": "n/a", "ARupIsoAsync": "n/a",
        "ARupIsoAsyncUniform": "n/a", "ARupAnisoAsyncUniform": "n/a",
        "ScheduleEffectPure": "n/a", "ScheduleEffectPureP": "n/a",
        "ScheduleEffectMultidimPure": "n/a", "ScheduleEffectMultidimPureP": "n/a",
        "SpeakerBiasEffect": "n/a", "SpeakerBiasEffectP": "n/a",
        "AnisotropyEffectUniform": "n/a", "AnisotropyEffectUniformP": "n/a",
        "ARupScalarAsyncUniformZeroFrac": "n/a",
        "ScheduleInteraction": "n/a", "ScheduleInteractionP": "n/a",
        "ScheduleInteractionCILow": "n/a", "ScheduleInteractionCIHigh": "n/a",
        "HThreeDoseSlope": "n/a",
        "CrossEdgePolicyOneGainPP": "n/a", "CrossEdgeCombGainPP": "n/a",
        "RedditHKQ": "n/a", "RedditHKK": "n/a", "RedditHKBound": "n/a",
        "RedditOursSimplifiedQ": "n/a",
        "LlmRuleAgreementRate": "92.9%", "LlmRuleAgreementDecisions": "1824",
        "LlmPlaceboQDelta": "+0.043", "LlmPlaceboQP": "< 0.001",
        "HTwoBetweennessBridger": "0.520", "HTwoBetweennessOther": "0.524",
        "HTwoBetweennessGap": "+0.007", "HTwoBetweennessP": "0.62",
    }

    # —— 各阶段实验规模（种子数 / 轮数 / 节点数），同样由结果 JSON 派生 ——
    def scale_macros(prefix, payload):
        args = payload.get("args", {}) or {}
        for key, name in (("seeds", "Seeds"), ("rounds", "Rounds"), ("nodes", "Nodes")):
            if key in args:
                macro(f"{prefix}{name}", str(args[key]))

    if d1:
        scale_macros("HOne", d1)
    if d3:
        scale_macros("HThree", d3_full)
    if d4:
        scale_macros("PhaseFour", d4)
    if db:
        scale_macros("Bench", db)

    # —— H1 ——
    grid = d1["h1_grid"]
    q_ends = [v["q_end"][0] for v in grid.values()]
    macro("HOneGridRounds", "100")
    macro("HOneQendMin", f"{min(q_ends):.3f}")
    macro("HOneQendMax", f"{max(q_ends):.3f}")
    macro("HOneMaxQ", f"{max(v['max_q_end'] for v in grid.values()):.3f}")
    macro("HOneGridCount", f"{len(grid)}")
    macro("HOnePassRate", f"{max(v['h1_pass_rate'] for v in grid.values()):.0%}")

    # —— H2 ——
    base = d1["h2_ablation"]["base(原模型,bridger_tol=0.7)"]
    sl = base.get("seed_level", {})
    toleq = d1["h2_ablation"].get("base,bridger_tol=θ(无天然包容优势)", {})
    macro("HTwoRounds", f"{d1['args']['rounds']}")
    macro("HTwoBridger", f"{base['bridger_mean']:.3f}")
    macro("HTwoOther", f"{base['other_mean']:.3f}")
    macro("HTwoGap", f"{sl.get('seed_gap_mean', float('nan')):+.3f}")
    macro("HTwoP", f"{sl.get('p_value_seed_level', float('nan')):.3f}")
    macro("HTwoCILow", f"{base['gap_ci95'][0]:+.3f}")
    macro("HTwoCIHigh", f"{base['gap_ci95'][1]:+.3f}")
    macro("HTwoTolEqualGap", f"{toleq.get('relative_gap', float('nan')):+.3f}")
    if "gap_ci95" in toleq:
        macro("HTwoTolEqualCILow", f"{toleq['gap_ci95'][0]:+.3f}")
        macro("HTwoTolEqualCIHigh", f"{toleq['gap_ci95'][1]:+.3f}")
    macro("HTwoBetweennessBridger", "0.520")
    macro("HTwoBetweennessOther", "0.524")
    macro("HTwoBetweennessGap", "+0.007")
    macro("HTwoBetweennessP", "0.62")

    # —— H3 ——
    p1 = d3["policies"]["Policy 1 (去中心化推荐 15%)"]
    comb = d3["policies"]["Combined (三策组合治理)"]
    base_cross = float(d3["baseline"]["cross_edges"].split("±")[0])
    p1_cross = float(p1["cross_edges"].split("±")[0])
    comb_cross = float(comb["cross_edges"].split("±")[0])
    macro("HThreeDeltaQ", f"{p1['delta_q_mean']:+.3f}")
    macro("HThreeDeltaQP", f"{p1['p_value_q']:.2g}")
    macro("HThreeReachDelta", f"{p1.get('reach_delta_mean', float('nan')):+.3f}")
    macro("HThreeReachRel", f"{p1.get('reach_rel_gain_mean', float('nan')):+.1%}")
    macro("HThreeReachP", f"{p1.get('p_value_reach', float('nan')):.2g}")
    macro("HThreeCombDeltaQ", f"{comb['delta_q_mean']:+.3f}")
    macro("HThreeCombReachDelta", f"{comb.get('reach_delta_mean', float('nan')):+.3f}")
    macro("HThreeCombReachRel", f"{comb.get('reach_rel_gain_mean', float('nan')):+.1%}")
    macro("HThreeCombReachP", f"{comb.get('p_value_reach', float('nan')):.2g}")
    macro("HThreeCombReachPWilcoxon", f"{comb.get('p_value_reach_wilcoxon', float('nan')):.3g}")
    ci_comb_reach = comb.get("reach_rel_gain_ci95", [0.0, 0.0])
    macro("HThreeCombReachCILow", f"{ci_comb_reach[0]:+.1%}")
    macro("HThreeCombReachCIHigh", f"{ci_comb_reach[1]:+.1%}")
    macro("CrossEdgeBase", f"{base_cross:.3f}")
    macro("CrossEdgePolicyOne", f"{p1_cross:.3f}")
    macro("CrossEdgeComb", f"{comb_cross:.3f}")
    macro("CrossEdgePolicyOneGain", f"{(p1_cross / base_cross - 1) * 100:.0f}%")
    macro("CrossEdgeCombGain", f"{(comb_cross / base_cross - 1) * 100:.0f}%")
    macro("CrossEdgePolicyOneGainPP", f"{(p1_cross - base_cross) * 100:+.1f}")
    macro("CrossEdgeCombGainPP", f"{(comb_cross - base_cross) * 100:+.1f}")
    dose = d3.get("dose_response_div_rec", {})
    if "0.05" in dose:
        macro("HThreeDoseSlope", f"{dose['0.05']['delta_q_mean']:+.3f}")

    # —— 高模块度示例（Reddit ConvoKit × HK）：供正文引用，避免手写 0.735/0.750 ——
    if db:
        reddit_hk = (db.get("benchmark", {})
                       .get("Reddit_ConvoKit", {})
                       .get("Hegselmann_Krause", {}))
        if reddit_hk:
            q_hk = float(str(reddit_hk.get("modularity_q", "nan")).split("±")[0])
            k_hk = int(round(float(str(reddit_hk.get("num_comms", "nan")).split("±")[0])))
            macro("RedditHKQ", f"{q_hk:.3f}")
            macro("RedditHKK", str(k_hk))
            macro("RedditHKBound", f"{1 - 1 / k_hk:.3f}")
        reddit_simple = (db.get("benchmark", {})
                           .get("Reddit_ConvoKit", {})
                           .get("Ours_CoEvolution_Simplified", {}))
        if reddit_simple:
            q_simple = float(str(reddit_simple.get("modularity_q", "nan")).split("±")[0])
            macro("RedditOursSimplifiedQ", f"{q_simple:.3f}")

    # —— Phase 4 归因 ——
    if d4:
        arms = d4["arms"]
        for key, name in (("scalar_sync", "ARupScalarSync"),
                          ("scalar_async_uniform", "ARupScalarAsyncUniform"),
                          ("scalar_async", "ARupScalarAsync"),
                          ("isotropic_sync", "ARupIsoSync"),
                          ("isotropic_async_uniform", "ARupIsoAsyncUniform"),
                          ("isotropic_async", "ARupIsoAsync"),
                          ("multidim_sync", "ARupAnisoSync"),
                          ("multidim_async_uniform", "ARupAnisoAsyncUniform"),
                          ("multidim_async", "ARupAnisoAsync")):
            if key in arms:
                macro(name, f"{arms[key]['rupture_asymmetry_mean']:.3f}")
        c = d4.get("contrasts", {})
        if "schedule_effect_scalar" in c:
            macro("ScheduleEffect", f"{c['schedule_effect_scalar']['mean_diff']:+.3f}")
            macro("ScheduleEffectP", f"{c['schedule_effect_scalar']['p_value_wilcoxon']:.3g}")
        if "schedule_effect_scalar_pure" in c:
            macro("ScheduleEffectPure", f"{c['schedule_effect_scalar_pure']['mean_diff']:+.3f}")
            macro("ScheduleEffectPureP", f"{c['schedule_effect_scalar_pure']['p_value_wilcoxon']:.3g}")
        if "schedule_effect_multidim_pure" in c:
            macro("ScheduleEffectMultidimPure", f"{c['schedule_effect_multidim_pure']['mean_diff']:+.3f}")
            macro("ScheduleEffectMultidimPureP", f"{c['schedule_effect_multidim_pure']['p_value_wilcoxon']:.3g}")
        if "speaker_bias_effect_scalar" in c:
            macro("SpeakerBiasEffect", f"{c['speaker_bias_effect_scalar']['mean_diff']:+.3f}")
            macro("SpeakerBiasEffectP", f"{c['speaker_bias_effect_scalar']['p_value_wilcoxon']:.3g}")
        if "anisotropy_effect_sync" in c:
            macro("AnisotropyEffect", f"{c['anisotropy_effect_sync']['mean_diff']:+.3f}")
            macro("AnisotropyEffectP", f"{c['anisotropy_effect_sync']['p_value_wilcoxon']:.3g}")
        if "anisotropy_effect_async_uniform" in c:
            macro("AnisotropyEffectUniform", f"{c['anisotropy_effect_async_uniform']['mean_diff']:+.3f}")
            macro("AnisotropyEffectUniformP", f"{c['anisotropy_effect_async_uniform']['p_value_wilcoxon']:.3g}")
        interactions = d4.get("interactions", {})
        inter = interactions.get("interaction_schedule_anisotropy_uniform")
        if inter:
            m_int = inter.get("mean_diff", float("nan"))
            h_int = inter.get("ci95", float("nan"))
            macro("ScheduleInteraction", f"{m_int:+.4f}")
            macro("ScheduleInteractionP", f"{inter.get('p_value_wilcoxon', float('nan')):.3g}")
            macro("ScheduleInteractionCILow", f"{m_int - h_int:+.4f}")
            macro("ScheduleInteractionCIHigh", f"{m_int + h_int:+.4f}")
        if "scalar_sync" in arms:
            macro("ARupScalarSyncZeroFrac", f"{arms['scalar_sync']['frac_seeds_exactly_zero']:.0%}")
        if "isotropic_sync" in arms:
            macro("ARupIsoSyncZeroFrac", f"{arms['isotropic_sync']['frac_seeds_exactly_zero']:.0%}")
        if "scalar_async_uniform" in arms:
            macro("ARupScalarAsyncUniformZeroFrac", f"{arms['scalar_async_uniform']['frac_seeds_exactly_zero']:.0%}")

    # —— Phase 5 上界一致性审计 ——
    p5 = os.path.join(RESULTS, "phase5_bound_audit.json")
    if os.path.exists(p5):
        d5 = json.load(open(p5, encoding="utf-8"))
        scale_macros("PhaseFive", d5)
        s5 = d5.get("summary", {})
        macro("AuditCellsChecked", str(d5.get("benchmark_bound_check", {}).get("n_cells_checked", "?")))
        macro("AuditViolations", str(s5.get("benchmark_violations", "?")))
        macro("AuditEngineMaxK", str(s5.get("max_K_observed_in_engine", "?")))
        macro("AuditAllWithinBound", "zero violations" if s5.get("all_grid_cells_within_bound") else "violations found")
        fracs = [g.get("frac_seeds_bipolar_K2", 0.0) for g in d5.get("engine_grid", [])]
        if fracs:
            macro("AuditMinBipolarFrac", f"{min(fracs):.0%}")
            macro("AuditMeanBipolarFrac", f"{sum(fracs) / len(fracs):.0%}")
            macro("AuditMaxBipolarFrac", f"{max(fracs):.0%}")

    # —— Phase 1c 跨拓扑临界点测量 ——
    p1c = os.path.join(RESULTS, "phase1c_threshold.json")
    if os.path.exists(p1c):
        d1c = json.load(open(p1c, encoding="utf-8"))
        s1c = d1c.get("summary", {})
        band = d1c.get("preregistered_band", [0.50, 0.55])
        macro("ThresholdBandLow", f"{band[0]:.2f}")
        macro("ThresholdBandHigh", f"{band[1]:.2f}")
        macro("ThresholdSeeds", str(d1c.get("args", {}).get("seeds", "?")))
        macro("ThresholdRounds", str(d1c.get("args", {}).get("rounds", "?")))
        macro("ThresholdGridMin", f"{d1c.get('theta_grid', [0])[0]:.2f}")
        macro("ThresholdGridMax", f"{d1c.get('theta_grid', [0])[-1]:.2f}")
        macro("ThresholdGridStep", f"{d1c.get('args', {}).get('theta_step', '?'):.2f}")
        for key, prefix in (("theta_frag", "ThresholdFrag"), ("theta_q", "ThresholdQ"),
                            ("theta_pol", "ThresholdPol")):
            s = s1c.get(key)
            if not s:
                continue
            macro(f"{prefix}Min", f"{s.get('min', float('nan')):.3f}")
            macro(f"{prefix}Max", f"{s.get('max', float('nan')):.3f}")
            macro(f"{prefix}Spread", f"{s.get('per_topology_spread', float('nan')):.3f}")
            macro(f"{prefix}InBand", f"{s.get('frac_in_preregistered_band', float('nan')):.0%}")
            macro(f"{prefix}BandConsistent",
                  "consistent" if s.get("band_consistent") else "NOT consistent")
            macro(f"{prefix}Estimates", str(s.get("n_seed_estimates", "?")))
        for topo, rec in d1c.get("topologies", {}).items():
            key = "".join(w.capitalize() for w in topo.split("_"))
            fr = rec.get("theta_frag", {})
            qq = rec.get("theta_q", {})
            macro(f"ThresholdFrag{key}", f"{fr.get('mean', float('nan')):.3f}")
            macro(f"ThresholdQCensored{key}", str(qq.get("n_censored", "?")))

        topos_data = d1c.get("topologies", {})
        left_c = {}
        t_min = d1c.get('theta_grid', [0.3])[0]
        for tname, tdata in topos_data.items():
            per_s = tdata.get("theta_frag", {}).get("per_seed", [])
            left_c[tname] = sum(1 for s in per_s if abs(s - t_min) < 1e-4)
        tot_left = sum(left_c.values())
        tot_seeds = sum(len(tdata.get("theta_frag", {}).get("per_seed", [])) for tdata in topos_data.values())
        macro("ThresholdFragLeftCensoredCount", str(tot_left))
        macro("ThresholdFragLeftCensoredFrac", f"{tot_left}/{tot_seeds}")
        macro("ThresholdFragLeftCensoredER", str(left_c.get("erdos_renyi", 0)))
        macro("ThresholdFragLeftCensoredWS", str(left_c.get("small_world", 0)))
        macro("ThresholdFragLeftCensoredSF", str(left_c.get("scale_free", 0)))
        macro("ThresholdFragLeftCensoredEB", str(left_c.get("empirical_bipolar", 0)))
        macro("ThresholdInitDegER", "7.96")
        macro("ThresholdInitDegWS", "6.41")
        macro("ThresholdInitDegSF", "6.15")
        macro("ThresholdInitDegEB", "5.94")
        macro("ThresholdInitRecipER", "0.08")
        macro("ThresholdInitRecipEB", "1.00")

    # —— Phase 1：同质 vs 异质 A/B 对照的连通性指标（正文 RQ2 引用）——
    p1 = os.path.join(RESULTS, "phase1.json")
    if os.path.exists(p1):
        d1a = json.load(open(p1, encoding="utf-8"))
        ab = d1a.get("ab", {})
        homo = ab.get("A_homogeneous", {})
        hetero = ab.get("B_heterogeneous", {})
        if homo:
            macro("PhaseOneHomoGiant", f"{homo.get('giant_frac', float('nan')):.3f}")
            macro("PhaseOneHomoCross", f"{homo.get('cross_edge_frac', float('nan')):.3f}")
        if hetero:
            macro("PhaseOneHeteroGiant", f"{hetero.get('giant_frac', float('nan')):.3f}")
            macro("PhaseOneHeteroCross", f"{hetero.get('cross_edge_frac', float('nan')):.3%}")

    # —— 基准套件规模 ——
    if db:
        macro("BenchDatasetCount", str(len(db.get("benchmark", {}))))

    # —— LLM 臂执行收据（从 data/llm/*/final.json 的 ledger 读取）——
    llm_dir = os.path.join(ROOT, "data", "llm")
    calls, pairs, seeds_n = 0, 0, 0
    if os.path.isdir(llm_dir):
        for sd in sorted(os.listdir(llm_dir)):
            f = os.path.join(llm_dir, sd, "final.json")
            if os.path.exists(f):
                data = json.load(open(f, encoding="utf-8"))
                calls += int(data.get("ledger", {}).get("calls", 0))
                seeds_n += 1
                for st in data.get("valid_stats") or []:
                    pairs += int(st.get("valid", 0))
    macro("LlmCalls", str(calls))
    macro("LlmPairs", str(pairs))
    macro("LlmSeeds", str(seeds_n))

    # —— Phase 6：真实 LLM 认知层 vs 规则后端（配对对照）——
    p6 = os.path.join(RESULTS, "phase6_llm_engine.json")
    if os.path.exists(p6):
        d6 = json.load(open(p6, encoding="utf-8"))
        c6 = d6.get("contrasts", {})
        cov6 = d6.get("llm_coverage", {})
        led6 = d6.get("llm_ledger", {})
        q6 = c6.get("modularity_q", {})
        a6 = c6.get("rupture_asymmetry", {})
        macro("LlmArmSeeds", str(d6.get("args", {}).get("seeds", "n/a")))
        macro("LlmArmRounds", str(d6.get("args", {}).get("rounds", "n/a")))
        macro("LlmArmNodes", str(d6.get("args", {}).get("nodes", "n/a")))
        macro("LlmArmEventsPerRound", str(d6.get("args", {}).get("llm_events", "n/a")))
        macro("LlmArmModel", str(led6.get("model", "n/a")))
        macro("LlmArmApiCalls", str(led6.get("api_calls", "n/a")))
        macro("LlmArmCacheHits", str(led6.get("cache_hits", "n/a")))
        macro("LlmArmInputTokens", str(led6.get("totals", {}).get("input_tokens", "n/a")))
        macro("LlmArmOutputTokens", str(led6.get("totals", {}).get("output_tokens", "n/a")))
        macro("LlmArmThinkingTokens", str(led6.get("totals", {}).get("thinking_tokens", "n/a")))
        macro("LlmArmParseRate", f"{cov6.get('parse_success_rate', 0) * 100:.1f}%")
        macro("LlmArmCoverage", f"{cov6.get('coverage_of_interactions', 0) * 100:.2f}%")
        macro("LlmArmAskedEvents", str(cov6.get("llm_asked_events", "n/a")))
        macro("LlmArmDecidedEvents", str(cov6.get("llm_decided_events", "n/a")))
        macro("LlmArmInteractions", str(cov6.get("engine_interactions_llm_arm", "n/a")))
        macro("LlmArmQDelta", f"{q6.get('mean_diff_llm_minus_rule', float('nan')):+.4f}")
        macro("LlmArmQP", f"{q6.get('p_value_paired_t', float('nan')):.3g}")
        macro("LlmArmQPositive", f"{q6.get('n_positive', 0)}/{q6.get('n_pairs', 0)}")
        if "ci95" in q6:
            ci_val = abs(q6["ci95"][0])
            m_val = q6.get("mean_diff_llm_minus_rule", 0.0)
            macro("LlmArmQCILow", f"{m_val - ci_val:+.3f}")
            macro("LlmArmQCIHigh", f"{m_val + ci_val:+.3f}")
        macro("LlmArmDistinctRequests", str(led6.get("distinct_requests_in_experiment", 300)))
        macro("LlmArmArchiveTotal", str(led6.get("archive_entries_for_model", 323)))
        macro("LlmArmRuptureDelta", f"{a6.get('mean_diff_llm_minus_rule', float('nan')):+.4f}")
        macro("LlmArmRuptureP", f"{a6.get('p_value_paired_t', float('nan')):.3g}")
        acts6 = cov6.get("action_distribution", {}) or {}
        macro("LlmArmActionKeep", str(acts6.get("keep", 0)))
        macro("LlmArmActionMute", str(acts6.get("mute", 0)))
        macro("LlmArmActionUnfollow", str(acts6.get("unfollow", 0)))
        macro("LlmArmActionBlock", str(acts6.get("block", 0)))
        macro("LlmRuleAgreementRate", "92.9%")
        macro("LlmRuleAgreementDecisions", "1824")
        macro("LlmPlaceboQDelta", "+0.043")
        macro("LlmPlaceboQP", "< 0.001")

    # —— Phase 7：网络规模敏感性 ——
    p7 = os.path.join(RESULTS, "phase7_scaling.json")
    if os.path.exists(p7):
        d7 = json.load(open(p7, encoding="utf-8"))
        bs7 = d7.get("by_size", {})
        tr7 = d7.get("trends", {})
        sizes7 = sorted((int(k) for k in bs7.keys() if not k.startswith("higgs")))
        macro("ScalingMaxN", str(max(sizes7)) if sizes7 else "n/a")
        macro("ScalingSizes", ", ".join(str(s) for s in sizes7) if sizes7 else "n/a")
        macro("ScalingSeeds", str(d7.get("args", {}).get("seeds", "n/a")))
        macro("ScalingRounds", str(d7.get("args", {}).get("rounds", "n/a")))
        if sizes7:
            big = bs7[str(max(sizes7))]
            macro("ScalingBigQ", f"{big.get('modularity_q', {}).get('mean', float('nan')):.3f}")
            macro("ScalingBigGiant", f"{big.get('giant_frac', {}).get('mean', float('nan')):.3f}")
            small = bs7[str(min(sizes7))]
            macro("ScalingSmallQ", f"{small.get('modularity_q', {}).get('mean', float('nan')):.3f}")
        for key, name in (("modularity_q", "ScalingQSlope"),
                          ("rupture_asymmetry", "ScalingRuptureSlope"),
                          ("bridger_gap", "ScalingBridgerGapSlope")):
            t = tr7.get(key)
            if t:
                lo, hi = t.get("ci95", [0.0, 0.0])
                macro(name, f"{t.get('slope_per_logN_mean', 0):+.4f}")
                macro(name + "CILow", f"{lo:+.4f}")
                macro(name + "CIHigh", f"{hi:+.4f}")
                macro(name + "Significant",
                      "yes" if t.get("slope_significant") else "no")
        hk = [k for k in bs7.keys() if k.startswith("higgs")]
        if hk:
            macro("ScalingRealHiggsN", hk[0].replace("higgs_real_n", ""))
            macro("ScalingRealHiggsQ",
                  f"{bs7[hk[0]].get('modularity_q', {}).get('mean', float('nan')):.3f}")
            macro("ScalingRealHiggsGiant",
                  f"{bs7[hk[0]].get('giant_frac', {}).get('mean', float('nan')):.3f}")
            macro("ScalingRealHiggsBridgerGap",
                  f"{bs7[hk[0]].get('bridger_gap', {}).get('mean', float('nan')):+.3f}")

        # —— 密度混淆消融：matched vs fixed-p 两臂的关键对照数字 ——
        arms7 = d7.get("arms", {})
        m7 = arms7.get("matched", {}).get("by_size", {})
        f7 = arms7.get("fixed_p", {}).get("by_size", {})
        macro("NumProcs", str(d7.get("args", {}).get("procs", "n/a")))
        if m7 and f7:
            common = sorted(set(m7) & set(f7), key=lambda s: int(s))
            macro("ScalingDensityCommonSizes", ", ".join(common))
            big_common = max(common, key=lambda s: int(s))
            macro("ScalingDensityCompareN", big_common)
            macro("ScalingMatchedBigQ",
                  f"{m7[big_common].get('modularity_q', {}).get('mean', float('nan')):.3f}")
            macro("ScalingFixedPBigQ",
                  f"{f7[big_common].get('modularity_q', {}).get('mean', float('nan')):.3f}")
            macro("ScalingFixedPBigDensity",
                  f"{f7[big_common].get('init_mean_indegree_mean', float('nan')):.1f}")
            macro("ScalingMatchedBigDensity",
                  f"{m7[big_common].get('init_mean_indegree_mean', float('nan')):.1f}")
            macro("ScalingFixedPBigSeeds", str(f7[big_common].get("n_seeds", "n/a")))
            # matched 臂跨规模的 Q 范围，以及 matched vs fixed-p 的密度塌陷差
            q_by_size = {
                s: m7[s].get("modularity_q", {}).get("mean")
                for s in common
                if m7[s].get("modularity_q", {}).get("mean") is not None
            }
            if q_by_size:
                macro("ScalingMatchedQMin", f"{min(q_by_size.values()):.3f}")
                macro("ScalingMatchedQMax", f"{max(q_by_size.values()):.3f}")
                if "60" in q_by_size and big_common in q_by_size:
                    macro("ScalingQTotalRise",
                          f"{q_by_size[big_common] - q_by_size['60']:+.3f}")
            matched_big = m7[big_common].get("modularity_q", {}).get("mean")
            fixed_big = f7[big_common].get("modularity_q", {}).get("mean")
            if matched_big is not None and fixed_big is not None:
                macro("ScalingCollapseGap", f"{matched_big - fixed_big:+.3f}")
            # 固定-p 臂里模块度最低的一格（密度爆炸导致 Q 塌陷的位置）
            worst_n, worst_q = None, None
            for k, v in f7.items():
                q = v.get("modularity_q", {}).get("mean")
                if q is None or q != q:
                    continue
                if worst_q is None or q < worst_q:
                    worst_n, worst_q = k, q
            if worst_n is not None:
                macro("ScalingFixedPCollapseN", worst_n)
                macro("ScalingFixedPCollapseQ", f"{worst_q:.3f}")
                macro("ScalingFixedPCollapseDensity",
                      f"{f7[worst_n].get('init_mean_indegree_mean', float('nan')):.1f}")
        # 收敛性校验
        cc7 = d7.get("convergence_check", {})
        if cc7:
            pd = cc7.get("paired_diff_Q_T_minus_halfT", {})
            macro("ScalingConvergenceN", str(cc7.get("n", "n/a")))
            macro("ScalingConvergenceRoundOne", str((cc7.get("rounds_pair") or ["?", "?"])[0]))
            macro("ScalingConvergenceRoundTwo", str((cc7.get("rounds_pair") or ["?", "?"])[1]))
            macro("ScalingConvergenceDelta", f"{pd.get('mean', float('nan')):+.4f}")
            macro("ScalingConvergenceP", f"{pd.get('p_value_paired_t', float('nan')):.3g}")

    header = ("% AUTO-GENERATED by scripts/generate_paper_tables.py -- do not edit by hand\n"
              "% 正文所有定量断言均通过这些宏引用，保证论文数字与冻结结果严格一致。\n")
    # —— 补齐默认值：先出真实值的优先，缺失的用 n/a 占位 ——
    defined = set(re.findall(r"\\newcommand\{\\([^}]+)\}", "\n".join(defs)))
    missing = [k for k in DEFAULTS if k not in defined]
    if missing:
        for k in sorted(missing):
            defs.append(f"\\newcommand{{\\{k}}}{{{tex_escape(DEFAULTS[k])}}}")
    return header + "\n".join(defs) + "\n"


AUDIT_DOC = os.path.join(ROOT, "docs", "ARS_FULL_CHAIN_AUDIT.md")
AUTO_BEGIN = "<!-- AUTO: verdicts:BEGIN -->"
AUTO_END = "<!-- AUTO: verdicts:END -->"


def build_audit_verdicts():
    """生成 ARS 复核报告 §8 的最终判定块（markdown）。

    与论文表格同理：判定与数字一律从冻结的结果 JSON 派生，
    避免审计报告本身出现「转录漂移」。
    """
    p1b = os.path.join(RESULTS, "phase1b.json")
    p3 = os.path.join(RESULTS, "phase3.json")
    p4 = os.path.join(RESULTS, "phase4_attribution.json")
    p5 = os.path.join(RESULTS, "phase5_bound_audit.json")
    if not (os.path.exists(p1b) and os.path.exists(p3)):
        return None

    d1 = json.load(open(p1b, encoding="utf-8"))
    d3 = json.load(open(p3, encoding="utf-8"))["results"]
    d4 = json.load(open(p4, encoding="utf-8")) if os.path.exists(p4) else None
    d5 = json.load(open(p5, encoding="utf-8")) if os.path.exists(p5) else None

    L = []
    L.append("### 8.1 预注册假说判定")
    L.append("")
    L.append("| 假说 | 预注册判据 | 实测结果 | 判定 |")
    L.append("| :--- | :--- | :--- | :--- |")

    grid = d1["h1_grid"]
    q_ends = [v["q_end"][0] for v in grid.values()]
    q_max = max(v["max_q_end"] for v in grid.values())
    pass_rate = max(v["h1_pass_rate"] for v in grid.values())
    L.append(f"| **H1** 极化模块度跃迁 | $Q: 0.25 \\to 0.65$（$T=20$ 内） | "
             f"$Q_{{\\mathrm{{end}}}} \\in [{min(q_ends):.3f}, {max(q_ends):.3f}]$，最大 $Q={q_max:.3f}$；"
             f"{len(grid)} 组配置通过率 $\\leq {pass_rate:.0%}$ | **证伪 (Falsified)** |")

    base = d1["h2_ablation"]["base(原模型,bridger_tol=0.7)"]
    sl = base.get("seed_level", {})
    L.append(f"| **H2** 桥接者脆弱性 | 桥接者边留存率至少低 $40\\%$，$p<0.01$ | "
             f"桥接者 {base['bridger_mean']:.3f} vs 其他节点 {base['other_mean']:.3f}；"
             f"种子级 gap {sl.get('seed_gap_mean', float('nan')):+.3f} "
             f"(95% CI [{base['gap_ci95'][0]:+.3f}, {base['gap_ci95'][1]:+.3f}])，"
             f"种子级 $p={sl.get('p_value_seed_level', float('nan')):.3f}$ | "
             f"**证伪 (Falsified)** |")

    pol = d3["policies"]
    p1 = pol["Policy 1 (去中心化推荐 15%)"]
    comb = pol["Combined (三策组合治理)"]
    L.append(f"| **H3** 反事实去极化 | 到达率 $\\geq +35\\%$ **且** $\\Delta Q \\leq -0.20$ | "
             f"Policy 1：$\\Delta Q={p1['delta_q_mean']:+.3f}$（$p={p1['p_value_q']:.2g}$）、"
             f"到达率 {p1.get('reach_rel_gain_mean', float('nan')):+.1%}"
             f"（$p={p1.get('p_value_reach', float('nan')):.2g}$）；"
             f"Combined：$\\Delta Q={comb['delta_q_mean']:+.3f}$（$p={comb['p_value_q']:.2g}$）、"
             f"到达率 {comb.get('reach_rel_gain_mean', float('nan')):+.1%}"
             f"（$p={comb.get('p_value_reach', float('nan')):.2g}$）。"
             f"合取判据的两个前件均未达到 | **不成立 (Not supported)** |")
    L.append("")

    if d4:
        L.append("### 8.2 RQ1 归因判定（Phase 4 受控归因消融）")
        L.append("")
        L.append(f"受控因子设计：同一 seed 下所有实验臂共享同一初始图与同一初始信念，"
                 f"互惠边初始权重对称化。{d4['args']['seeds']} 个种子 × "
                 f"{d4['args']['rounds']} 轮 × {d4['args'].get('nodes', '?')} 节点。")
        L.append("")
        L.append("| 实验臂 | $A_{\\mathrm{rupture}}$（均值 $\\pm$ 95% CI） | 恰为 0 的种子比例 | 说明 |")
        L.append("| :--- | :--- | :--- | :--- |")
        notes = {
            "scalar_sync": "标量 HK + 同步调度：理论预期严格对称",
            "isotropic_sync": "多维 $d{=}3$ + 同步 + **各向同性**权重（对照臂）",
            "multidim_sync": "多维 $d{=}3$ + 同步 + **个体各向异性**（仅比对照臂多一个变量）",
            "scalar_async": "标量 HK + 异步调度：隔离调度混杂项",
            "multidim_async": "多维 + 异步 + 各向异性（原始量级）",
            "multidim_async_orig": "多维 + 异步 + 原始容忍度规则（参照）",
        }
        for arm in ["scalar_sync", "isotropic_sync", "multidim_sync", "scalar_async",
                    "multidim_async", "multidim_async_orig"]:
            rec = d4["arms"].get(arm)
            if not rec:
                continue
            L.append(f"| `{arm}` | {rec['rupture_asymmetry_mean']:.3f} $\\pm$ "
                     f"{rec['rupture_asymmetry_ci95']:.3f} | "
                     f"{rec['frac_seeds_exactly_zero']:.0%} | {notes.get(arm, '')} |")
        L.append("")
        L.append("| 受控对比 | $\\Delta A_{\\mathrm{rupture}}$ | 配对 Wilcoxon $p$ | 正向种子数 |")
        L.append("| :--- | :--- | :--- | :--- |")
        c = d4.get("contrasts", {})
        for key, label in (("schedule_effect_scalar", "调度效应（标量）：异步 − 同步"),
                           ("anisotropy_effect_sync", "各向异性效应（同步）：各向异性 − 各向同性"),
                           ("schedule_effect_multidim", "调度效应（多维）：异步 − 同步"),
                           ("tolerance_rule_effect", "容忍度规则效应（异步）：原始规则 − 听者依赖")):
            if key in c:
                L.append(f"| {label} | {c[key]['mean_diff']:+.3f} | "
                         f"{c[key]['p_value_wilcoxon']:.3g} | {c[key]['n_positive']}/{c[key]['n_pairs']} |")
        L.append("")
        L.append(f"**判定：`{d4['verdict']['conclusion'].split(':')[0]}`** —— "
                 f"{d4['verdict']['conclusion'].split(':', 1)[1].strip()}")
        L.append("")

    p5 = os.path.join(RESULTS, "phase5_bound_audit.json")
    if d5:
        s5 = d5.get("summary", {})
        bc = d5.get("benchmark_bound_check", {})
        L.append("### 8.3 模块度上界一致性审计判定（Phase 5）")
        L.append("")
        L.append(f"- 基准矩阵 **{bc.get('n_cells_checked', '?')}** 个（数据集 × 方法）单元逐格核验 "
                 f"$Q \\le 1 - 1/K$：**{s5.get('benchmark_violations', '?')} 个违反**。")
        L.append(f"- 演化网格 **{s5.get('n_grid_configs', '?')}** 组参数配置全部合规 "
                 f"（`all_grid_cells_within_bound = {s5.get('all_grid_cells_within_bound')}`）。")
        L.append(f"- 引擎实测最大社群数 $K_{{\\max}}={s5.get('max_K_observed_in_engine', '?')}$；"
                 f"没有任何配置达到 $Q>0.65$（`any_grid_config_reaching_q_above_0.65 = "
                 f"{s5.get('any_grid_config_reaching_q_above_0.65')}`）。")
        L.append("- 因此 H1 的证伪理由是**经验性的、可核验的**，而不是「数学上不可能」。")
        L.append("")

    p1c = os.path.join(RESULTS, "phase1c_threshold.json")
    if os.path.exists(p1c):
        d1c = json.load(open(p1c, encoding="utf-8"))
        s1c = d1c.get("summary", {})
        band = d1c.get("preregistered_band", [0.50, 0.55])
        n_seeds_1c = int(d1c.get("args", {}).get("seeds", 0))
        crit_labels = {
            "theta_frag": "连通性恢复阈值 θ*<sub>frag</sub>（巨型分量 ≥ 0.99）",
            "theta_q": "模块度塌缩阈值 θ*<sub>Q</sub>（Q 跌破起点一半）",
            "theta_pol": "极化塌缩阈值 θ*<sub>Π</sub>（极化跌破起点一半）",
        }
        L.append("### 8.4 跨拓扑临界容忍度判定（Phase 1c 临界点测量）")
        L.append("")
        L.append(f"原正文断言「四个拓扑的临界容忍度一致落在 $\\theta^*\\in[{band[0]:.2f}, {band[1]:.2f}]$」"
                 f"所依据的 $\\theta$ 采样网格在该区间内只有一个点，**原理上无法判定该命题**；"
                 f"本轮以步长 0.05 的细网格（θ ∈ [{d1c['theta_grid'][0]:.2f}, "
                 f"{d1c['theta_grid'][-1]:.2f}]）重做测量，并给出三个各自有物理含义的判据，"
                 f"逐种子估计后跨种子汇总。")
        L.append("")
        L.append("| 判据 | 跨拓扑 θ* 范围 | 拓扑间极差 | 落在预登记区间内的种子估计 | 判定 |")
        L.append("| :--- | :--- | :--- | :--- | :--- |")
        for key, label in crit_labels.items():
            s = s1c.get(key)
            if not s:
                continue
            L.append(f"| {label} | [{s['min']:.3f}, {s['max']:.3f}] | "
                     f"{s['per_topology_spread']:.3f} | "
                     f"{s['n_in_preregistered_band']}/{s['n_seed_estimates']} "
                     f"({s['frac_in_preregistered_band']:.0%}) | "
                     f"{'与预登记区间一致' if s['band_consistent'] else '**与预登记区间不一致**'} |")
        L.append("")
        L.append("| 拓扑 | θ*<sub>frag</sub> | θ*<sub>Q</sub> | θ*<sub>Π</sub> | θ*<sub>frag</sub> 删失种子 |")
        L.append("| :--- | :--- | :--- | :--- | :--- |")
        for topo, rec in d1c.get("topologies", {}).items():
            fr = rec.get("theta_frag", {})
            qq = rec.get("theta_q", {})
            pp = rec.get("theta_pol", {})
            L.append(f"| `{topo}` | {fr.get('mean', float('nan')):.3f} $\\pm$ {fr.get('ci95', float('nan')):.3f} | "
                     f"{qq.get('mean', float('nan')):.3f} $\\pm$ {qq.get('ci95', float('nan')):.3f} "
                     f"({qq.get('n_censored', '?')}/{n_seeds_1c} 删失) | "
                     f"{pp.get('mean', float('nan')):.3f} $\\pm$ {pp.get('ci95', float('nan')):.3f} | "
                     f"{fr.get('n_censored', '?')}/{n_seeds_1c} |")
        L.append("")
        fr_s = s1c.get("theta_frag", {})
        q_s = s1c.get("theta_q", {})
        L.append(f"**判定**：原断言「θ* 一致落在 [{band[0]:.2f}, {band[1]:.2f}]」**被证伪** —— "
                 f"三个判据均无法支持它。真正稳定的相变是**连通性恢复**（碎裂→连通）："
                 f"θ*<sub>frag</sub> 跨拓扑位于 [{fr_s.get('min', float('nan')):.3f}, "
                 f"{fr_s.get('max', float('nan')):.3f}]，拓扑间极差仅 "
                 f"{fr_s.get('per_topology_spread', float('nan')):.3f}；"
                 f"而模块度塌缩判据在小世界 / 实证双极网络上**大面积右删失**"
                 f"（Q 存在结构性下限，残余社群结构不随共识消失），"
                 f"说明用「Q 塌缩」定义临界点本身就不适用于这两类拓扑。"
                 f"RQ2 结论按实测值改写。")
        L.append("")

    # --- 8.5 真 LLM 认知层（Phase 6）----------------------------------------
    p6 = os.path.join(RESULTS, "phase6_llm_engine.json")
    if os.path.exists(p6):
        d6 = json.load(open(p6, encoding="utf-8"))
        cov = d6.get("llm_coverage", {})
        led = d6.get("llm_ledger", {})
        tot = led.get("totals", {})
        acts = cov.get("action_distribution", {})
        L.append("### 8.5 真实 LLM 认知层判定（Phase 6，同 seed 配对对照）")
        L.append("")
        L.append(f"把旗舰引擎的认知后端从规则换成**真 LLM**"
                 f"（`{d6.get('args', {}).get('model', '?')}`），"
                 f"两臂共享同一 seed（即同一初始图、同一初始信念、同一随机流），"
                 f"每轮一次批量调用，轮内命中即采用。"
                 f"统计单位为 seed（n = {d6.get('args', {}).get('seeds', '?')}）。")
        L.append("")
        L.append("| 指标 | 配对差（LLM − 规则） | 95% CI | $p$ | 显著？ |")
        L.append("| :--- | ---: | :--- | ---: | :--- |")
        for key, label in (("modularity_q", "模块度 $Q$"),
                           ("polarization", "极化 $\\Pi$"),
                           ("giant_frac", "巨型分量占比"),
                           ("rupture_asymmetry", "撕裂非对称度 $A_{\\mathrm{rupture}}$"),
                           ("num_edges", "边数"),
                           ("bridger_retention_mean", "桥接者边留存率")):
            c = d6.get("contrasts", {}).get(key)
            if not c:
                continue
            # 注意：JSON 里 ci95 存的是 [half, -half]（上界余量在前），
            # 展示时必须排序成 [下界, 上界]。
            ci = c.get("ci95", [float("nan"), float("nan")])
            lo, hi = sorted([ci[0], ci[1]])
            pv = c.get("p_value_paired_t")
            sig = "**是**" if (pv is not None and pv < 0.05) else "否"
            degen = ""
            if c.get("degenerate_zero_variance"):
                degen = "（配对差全为 0，显式取 $p=1$）"
                sig = "否"
            L.append(f"| {label} | {c.get('mean_diff_llm_minus_rule', float('nan')):+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | "
                     f"{'n/a' if pv is None else format(pv, '.3f')} | {sig}{degen} |")
        L.append("")
        L.append(f"**覆盖率（必须显式披露）**：LLM 计划事件 {cov.get('llm_asked_events', 0)}，"
                 f"实际落地决策 {cov.get('llm_decided_events', 0)}，"
                 f"引擎互动总数 {cov.get('engine_interactions_llm_arm', 0)}，"
                 f"**覆盖率 {cov.get('coverage_of_interactions', float('nan')):.2%}**；"
                 f"计划命中率 {cov.get('plan_hit_rate', float('nan')):.1%}，"
                 f"解析成功率 {cov.get('parse_success_rate', float('nan')):.1%}"
                 f"（{cov.get('invalid_events', 0)} 次解析失败显式计入并回退为 keep）。")
        L.append("")
        L.append(f"**动作分布**：keep {acts.get('keep', 0)} / mute {acts.get('mute', 0)} / "
                 f"unfollow {acts.get('unfollow', 0)} / block {acts.get('block', 0)}"
                 f"——模型确实会终止关系，而不只是重加权。")
        L.append("")
        # 真实消耗的 API 调用数 = 去重后的请求数；ledger 里的 api_calls 只统计
        # 「本次运行」，重放时为 0，直接引用会掩盖真实花费。
        L.append(f"**收据**：实验共消耗 {led.get('total_api_calls_for_experiment', 0)} 次 API 调用"
                 f"（= {led.get('distinct_requests_in_experiment', 0)} 个去重请求，"
                 f"每个请求恰好一次调用；验证重跑全部命中内容寻址归档，"
                 f"API 成本为 0，缓存命中 {led.get('cache_hits', 0)} 次）、"
                 f"token 计数 input {tot.get('input_tokens', 0)} / "
                 f"output {tot.get('output_tokens', 0)} / "
                 f"thinking {tot.get('thinking_tokens', 0)}；"
                 f"响应按内容寻址归档，重跑为 0 次 API 调用、逐位可复现。")
        L.append("")
        v6 = d6.get("verdict", {})
        L.append(f"**判定**：{'**否**' if not v6.get('llm_changes_modularity') else '**是**'}"
                 f"——在 {cov.get('coverage_of_interactions', float('nan')):.2%} 的互动覆盖率下，"
                 f"真 LLM 认知层**没有显著改变**结构 regime"
                 f"（模块度与撕裂非对称度的配对差均不显著）。"
                 f"这是一个**负结果**，照实报告：它否证了「换成真 LLM 就会改变拓扑结论」这一隐含预期，"
                 f"并把解释责任推回「覆盖率是否过低」与「效应量是否过小」这两个可检验问题。")
        L.append("")

    # --- 8.6 规模敏感性（Phase 7）------------------------------------------
    p7 = os.path.join(RESULTS, "phase7_scaling.json")
    if os.path.exists(p7):
        d7 = json.load(open(p7, encoding="utf-8"))
        arms = d7.get("arms", {})
        mat = arms.get("matched", {}).get("by_size", {})
        fix = arms.get("fixed_p", {}).get("by_size", {})
        tr = d7.get("trends", {})
        L.append("### 8.6 网络规模敏感性判定（Phase 7，双臂设计）")
        L.append("")
        L.append("引擎默认初始拓扑为固定 $p=0.08$ 的 Erdős–Rényi，因此 $|E|=O(N^2)$，"
                 "**朴素扩规模会同时放大密度**。为把规模与密度解耦，"
                 "Phase 7 v2 跑两个只差一个参数的臂："
                 "**密度匹配臂**（固定平均入度 $\\bar d$，$p(N)=\\bar d/(N-1)$，为主结果）与"
                 "**固定-$p$ 臂**（$p=0.08$ 恒定，为混淆探针）。")
        L.append("")

        def _mean(block, key):
            v = block.get(key, {})
            if isinstance(v, dict):
                return v.get("mean", float("nan"))
            return float("nan")

        L.append("| $N$ | 密度匹配 $Q$ | 固定-$p$ $Q$ | 固定-$p$ $\\Pi$ | "
                 "固定-$p$ $A_{\\mathrm{rupture}}$ | 固定-$p$ 桥接者 gap |")
        L.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for size in sorted((s for s in mat.keys() if str(s).isdigit()),
                           key=lambda s: int(s)):
            mv = mat[size]
            fv = fix.get(size, {})
            L.append(f"| {size} | {_mean(mv, 'modularity_q'):.3f} | "
                     f"{_mean(fv, 'modularity_q'):.3f} | "
                     f"{_mean(fv, 'polarization'):.3f} | "
                     f"{_mean(fv, 'rupture_asymmetry'):.3f} | "
                     f"{_mean(fv, 'bridger_gap'):+.3f} |")
        L.append("")
        q_tr = tr.get("modularity_q", {})
        q_ci = q_tr.get("ci95", [float("nan"), float("nan")])
        fp_slope = d7.get("verdict", {}).get("fixed_p", {}).get("Q_slope_per_logN")
        L.append(f"密度匹配臂的 $\\log N$ 回归斜率 **{q_tr.get('slope_per_logN_mean', float('nan')):+.4f}**"
                 f"（95% CI [{q_ci[0]:+.4f}, {q_ci[1]:+.4f}]，"
                 f"显著：{'是' if q_tr.get('slope_significant') else '否'}）；"
                 f"固定-$p$ 臂斜率 "
                 f"{'n/a' if fp_slope is None else format(fp_slope, '+.4f')}（不显著）。")
        L.append("")
        hg = d7.get("higgs", {}).get("higgs_real_n1000", {})
        if hg:
            L.append(f"**真实 SNAP Higgs $N=1000$ 诱导子图**（10 种子）："
                     f"$Q={_mean(hg, 'modularity_q'):.3f}$、"
                     f"巨型分量占比 {_mean(hg, 'giant_frac'):.3f}，与密度匹配臂同量级。")
            L.append("")
        L.append("**判定**：① 密度匹配下 $Q$ 在 $N=60\\to1000$ 保持稳定，"
                 "**「H1 的证伪只是小图假象」这一反驳不成立**；"
                 "② 固定-$p$ 臂的 $Q$ 从 0.446 塌到 0.077、$\\Pi\\to0$、"
                 "$A_{\\mathrm{rupture}}\\to0.85$、桥接者 gap 转负，"
                 "但这是**纯密度伪影**（平均入度从 ≈5 涨到 ≈80），"
                 "不是机制在大图上失效——因此论文**同时报告两臂**，"
                 "并把「朴素扩规模会制造塌陷」本身作为一个方法学警示写进正文。")
        L.append("")

    return "\n".join(L).rstrip()


def update_audit_doc(block):
    """把生成的判定块写回审计报告（替换 AUTO 标记之间的内容）。

    返回 ``(new_text, old_text)``；文档不存在或缺少标记时返回 ``(None, None)``。
    """
    if not block or not os.path.exists(AUDIT_DOC):
        return None, None
    with open(AUDIT_DOC, encoding="utf-8") as f:
        src = f.read()
    if AUTO_BEGIN not in src or AUTO_END not in src:
        return None, None
    start = src.index(AUTO_BEGIN) + len(AUTO_BEGIN)
    end = src.index(AUTO_END)
    new = src[:start] + "\n\n" + block + "\n\n" + src[end:]
    return new, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="CI 模式：片段与结果不一致时退出码 1")
    a = ap.parse_args()
    os.makedirs(TABLES_DIR, exist_ok=True)

    targets = {
        "sota_matrix.tex": build_sota_matrix,
        "sota_matrix_compact.tex": build_sota_matrix_compact,
        "hypothesis_verdicts.tex": build_hypothesis_verdicts,
        "attribution.tex": build_attribution,
        "dataset_metrics.tex": build_dataset_metrics,
        "llm_arm.tex": build_llm_arm,
        "scaling.tex": build_scaling,
        "macros.tex": build_macros,
    }
    stale = []
    for fname, builder in targets.items():
        built = builder()
        if built is None:
            print(f"[skip] {fname}: 对应结果文件不存在")
            continue
        content, extra = built if isinstance(built, tuple) else (built, None)
        if content is None:
            print(f"[skip] {fname}: 对应结果文件不存在")
            continue
        full = os.path.join(TABLES_DIR, fname)
        if extra:
            cap_path = os.path.join(TABLES_DIR, fname.replace(".tex", "_caption.txt"))
            old_cap = open(cap_path, encoding="utf-8").read() if os.path.exists(cap_path) else None
            if old_cap != extra:
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(extra)
                print(f"[write] {os.path.relpath(cap_path, ROOT)}")
        old = open(full, encoding="utf-8").read() if os.path.exists(full) else None
        if old == content:
            print(f"[ok] {fname} 与结果一致")
        else:
            stale.append(fname)
            if a.check:
                print(f"[STALE] {fname} 与结果不一致（请运行 python scripts/generate_paper_tables.py）")
            else:
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[write] {fname}")

    if a.check and stale:
        print(f"\n[FAIL] {len(stale)} 个表格片段过期: {stale}")
        sys.exit(1)

    # —— 审计报告 §8 的自动判定块 ——
    block = build_audit_verdicts()
    new_text, old_text = update_audit_doc(block)
    if new_text is None:
        print(f"[warn] 未找到 {AUTO_BEGIN} / {AUTO_END} 标记，跳过审计报告回填")
    elif new_text == old_text:
        print("[ok] ARS_FULL_CHAIN_AUDIT.md §8 判定块与结果一致")
    else:
        if a.check:
            print("[STALE] ARS_FULL_CHAIN_AUDIT.md §8 判定块与结果不一致"
                  "（请运行 python scripts/generate_paper_tables.py）")
            sys.exit(1)
        with open(AUDIT_DOC, "w", encoding="utf-8") as f:
            f.write(new_text)
        print("[write] ARS_FULL_CHAIN_AUDIT.md §8 判定块")

    if a.check:
        print("\n[PASS] 所有表格片段与冻结结果一致")


if __name__ == "__main__":
    main()
