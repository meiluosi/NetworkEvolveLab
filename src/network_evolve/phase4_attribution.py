"""Phase 4：非对称撕裂（Asymmetric Rupture）机制归因消融 —— RQ1 的决定性实验。

背景
----
原论文把 A_rupture（互惠二元组中仅单向断裂的比例）归因于「多维语义认知」。
但项目的自有数据同时显示：标量 HK 模型在 Phase 2 离线臂里也出现了非零 A_rupture
（eps=0.6 -> 0.029，eps=1.2 -> 0.498），而经典有界信任模型的断裂判据在理论上是对称的。
这说明存在一个未被控制的混杂变量：**异步单发言者调度**。
只有把「调度方式」与「语义维度/各向异性」拆成正交因子，才能判定 A_rupture 的真正来源。

因子设计（受控：同一 seed 下所有实验臂共享完全相同的初始图与初始信念）
--------------------------------------------------------------------
  因子 A：更新调度   update_mode ∈ {async（随机单发言者）, sync（全体同步）}
  因子 B：认知几何   {scalar（d=1, HK 标量有界信任）, multidim（d=3, 代理语义 + 个体议题权重）}
  附加对照：isotropic（d=3 但全体等权、判据严格对称）→ 剥离「个体各向异性」这一变量

实验臂
------
  scalar_async        : 标量 + 异步          （调度混杂项）
  scalar_sync         : 标量 + 同步          （经典对称模型，理论预期 A ≡ 0）
  multidim_async      : 多维 + 异步 + 各向异性容忍
  multidim_sync       : 多维 + 同步 + 各向异性容忍   （仅相对 isotropic_sync 差在「个体各向异性」）
  isotropic_sync      : 多维 + 同步 + 各向同性（严格对称判据）
  multidim_async_orig : 多维 + 异步 + 原始容忍度规则（原始模型参照）

判读
----
  · 若 scalar_sync 与 isotropic_sync 的 A 严格为 0，而 multidim_sync > 0
    => 非对称撕裂来自「个体各向异性语义判据」（支持原假说，但需限定表述）
  · 若 scalar_async > 0 而 scalar_sync = 0
    => 非对称撕裂部分来自「异步调度」，原表述属过度归因，必须收窄

用法: python -m src.network_evolve.phase4_attribution --seeds 20 --rounds 30 --out results/phase4_attribution.json
"""
import argparse
import json
import numpy as np
from multiprocessing import Pool
from scipy import stats

from .semantic import VecWorld
from .expkit import mean_ci, fmt

# —— 实验臂定义：每个臂只与对照臂相差一个因子 ——
ARMS = {
    "scalar_async": dict(d=1, backend="hk", update_mode="async", symmetric_tol=False, isotropic=True),
    "scalar_sync": dict(d=1, backend="hk", update_mode="sync", symmetric_tol=False, isotropic=True),
    "multidim_async": dict(d=3, backend="proxy", update_mode="async", symmetric_tol=True, isotropic=False),
    "multidim_sync": dict(d=3, backend="proxy", update_mode="sync", symmetric_tol=True, isotropic=False),
    "isotropic_sync": dict(d=3, backend="proxy", update_mode="sync", symmetric_tol=True, isotropic=True),
    "multidim_async_orig": dict(d=3, backend="proxy", update_mode="async", symmetric_tol=False, isotropic=False),
}

ARM_LABELS = {
    "scalar_async": "Scalar d=1 / async",
    "scalar_sync": "Scalar d=1 / sync",
    "multidim_async": "Multi-d=3 / async / anisotropic",
    "multidim_sync": "Multi-d=3 / sync / anisotropic",
    "isotropic_sync": "Multi-d=3 / sync / isotropic",
    "multidim_async_orig": "Multi-d=3 / async / original tol (reference)",
}


def canonical_conditions(seed: int, n: int = 60, d: int = 3, p_edge: float = 0.15):
    """该 seed 的规范初始条件：初始图 g0、3 维信念 b3、1 维投影 b1、个体议题权重 salience。

    标量臂使用 b3 的第一维投影，从而与多维臂共享同一阵营结构与同一张初始图，
    保证不同臂之间的对比是「单因子受控对比」而非「不同抽样」。

    互惠二元组的初始权重被强制对称化（w_uv = w_vu）。原因：若互惠两边的初始强度不同，
    即使两个方向的断裂判据完全相同，较弱的一侧也会先断连，从而在指标上表现为「非对称断裂」。
    对称化初始权重后，A_rupture 只反映「断裂决策本身是否对称」，从而把语义/调度机制
    与「初始关系强度差异」这一混杂变量彻底分离。
    """
    base = VecWorld(n=n, d=d, backend="hk", seed=seed, p_edge=p_edge, p_form=0.1)
    g0 = base.g
    # 互惠边权重对称化
    for u, v in list(g0.edges()):
        if (v, u) in g0.edges() and u < v:
            w = 0.5 * (g0[u][v]["weight"] + g0[v][u]["weight"])
            g0[u][v]["weight"] = w
            g0[v][u]["weight"] = w
    return g0, base.b.copy(), base.b[:, [0]].copy(), base.salience.copy()


def initial_weight_asymmetry(g) -> float:
    """互惠二元组的初始权重不对称度：mean |w_uv - w_vu| / mean(w)（受控后应约为 0）。"""
    diffs = []
    for u, v in g.edges():
        if (v, u) in g.edges() and u < v:
            diffs.append(abs(g[u][v]["weight"] - g[v][u]["weight"]))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def run_arm(args):
    arm_name, seed, rounds, n, p_edge = args
    cfg = ARMS[arm_name]
    d = cfg["d"]
    g0, b3, b1, sal = canonical_conditions(seed, n=n, d=3, p_edge=p_edge)
    b_init = b1 if d == 1 else b3
    sal_init = None if d == 1 else (None if cfg["isotropic"] else sal)
    w = VecWorld(n=n, d=d, backend=cfg["backend"], eps=0.6, seed=seed + 1000,
                 p_edge=p_edge, p_form=0.1, update_mode=cfg["update_mode"],
                 isotropic=cfg["isotropic"], symmetric_tol=cfg["symmetric_tol"],
                 b_init=b_init, g0=g0, salience_init=sal_init)
    for _ in range(rounds):
        if w.update_mode == "sync":
            w._step_sync()
        else:
            for _ in range(w.n):
                w.step()
        w.closure()
    m = w.metrics()
    m["arm"] = arm_name
    m["seed"] = seed
    m["initial_weight_asymmetry"] = initial_weight_asymmetry(g0)
    return m


def _paired_contrast(a_runs, b_runs, label):
    """配对对比 A - B（同一 seed 配对），返回效应量与检验。"""
    a_by_seed = {r["seed"]: r["rupture_asymmetry"] for r in a_runs}
    b_by_seed = {r["seed"]: r["rupture_asymmetry"] for r in b_runs}
    diffs, seeds = [], []
    for s in sorted(set(a_by_seed) & set(b_by_seed)):
        va, vb = a_by_seed[s], b_by_seed[s]
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
            seeds.append(s)
    diffs = np.array(diffs)
    out = {"contrast": label, "n_pairs": int(len(diffs))}
    if len(diffs) < 2:
        out.update({"mean_diff": float("nan"), "p_value_wilcoxon": float("nan"),
                    "p_value_paired_t": float("nan"), "n_both_zero": 0})
        return out
    try:
        p_w = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue) if np.any(diffs != 0) else 1.0
    except ValueError:
        p_w = 1.0
    p_t = float(stats.ttest_rel([a_by_seed[s] for s in seeds], [b_by_seed[s] for s in seeds]).pvalue)
    out.update({
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "p_value_wilcoxon": p_w,
        "p_value_paired_t": p_t,
        "n_positive": int(np.sum(diffs > 0)),
        "n_both_zero": int(np.sum(diffs == 0)),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 4: Asymmetric-rupture mechanism attribution ablation")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--p-edge", type=float, default=0.15)
    ap.add_argument("--out", default="results/phase4_attribution.json")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    log = []

    def P(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        log.append(line)

    P(f"=== Phase 4 归因消融 (seeds={a.seeds}, rounds={a.rounds}, nodes={a.nodes}, p_edge={a.p_edge}) ===")
    P("所有实验臂共享同一 seed 的初始图与初始信念（单因子受控对比）；互惠边初始权重已对称化")

    jobs = [(arm, s, a.rounds, a.nodes, a.p_edge) for arm in ARMS for s in seeds]
    with Pool(a.procs) as pool:
        raw = pool.map(run_arm, jobs)

    by_arm = {arm: [r for r in raw if r["arm"] == arm] for arm in ARMS}

    out = {"args": vars(a), "arms": {}, "contrasts": {}}

    P("\narm | A_rupture mean±CI | frac(A≡0) | frac(A未定义) | n_mutual_broken | polarization | modularity")
    for arm, runs in by_arm.items():
        asym = np.array([r["rupture_asymmetry"] for r in runs], dtype=float)
        defined = asym[np.isfinite(asym)]
        frac_zero = float(np.mean(np.isclose(defined, 0.0))) if len(defined) else float("nan")
        frac_undef = float(np.mean(~np.isfinite(asym))) if len(asym) else float("nan")
        m, h = mean_ci(defined) if len(defined) else (float("nan"), float("nan"))
        rec = {
            "label": ARM_LABELS[arm],
            "rupture_asymmetry_mean": float(m),
            "rupture_asymmetry_ci95": float(h),
            "rupture_asymmetry_per_seed": [None if not np.isfinite(x) else float(x) for x in asym],
            "frac_seeds_exactly_zero": frac_zero,
            "frac_seeds_undefined": frac_undef,
            "n_mutual_broken": fmt([r["n_mutual_broken"] for r in runs]),
            "n_one_sided": fmt([r["n_one_sided"] for r in runs]),
            "polarization": fmt([r["polarization"] for r in runs]),
            "modularity": fmt([r["modularity"] for r in runs]),
            "small_comm_frac": fmt([r["small_comm_frac"] for r in runs]),
            "cross_camp_edge_frac": fmt([r["cross_camp_edge_frac"] for r in runs]),
            "edges": fmt([r["edges"] for r in runs]),
            "num_comms": fmt([r["num_comms"] for r in runs]),
            "initial_weight_asymmetry": fmt([r["initial_weight_asymmetry"] for r in runs]),
        }
        out["arms"][arm] = rec
        P(f"{ARM_LABELS[arm]:<42} | {m:.3f}±{h:.3f} | {frac_zero:.2f} | {frac_undef:.2f} | "
          f"{rec['n_mutual_broken']} | {rec['polarization']} | {rec['modularity']}")

    P("\n=== 决定性对比（配对，同 seed）===")
    contrasts = [
        ("schedule_effect_scalar", "scalar_async", "scalar_sync",
         "调度效应（标量）：异步 - 同步"),
        ("schedule_effect_multidim", "multidim_async", "multidim_sync",
         "调度效应（多维）：异步 - 同步"),
        ("anisotropy_effect_sync", "multidim_sync", "isotropic_sync",
         "个体各向异性效应（同步）：各向异性 - 各向同性"),
        ("tolerance_rule_effect", "multidim_async_orig", "multidim_async",
         "容忍度规则效应（异步）：原始规则 - 听者依赖规则"),
    ]
    for key, arm_a, arm_b, label in contrasts:
        c = _paired_contrast(by_arm[arm_a], by_arm[arm_b], label)
        out["contrasts"][key] = c
        P(f"{label}: mean ΔA = {c['mean_diff']:+.4f} (n={c['n_pairs']}, "
          f"pos={c.get('n_positive', 0)}, both-zero={c.get('n_both_zero', 0)}) "
          f"wilcoxon p={c['p_value_wilcoxon']:.3g}")

    # —— 归因结论判定 ——
    scalar_sync = out["arms"]["scalar_sync"]
    iso_sync = out["arms"]["isotropic_sync"]
    md_sync = out["arms"]["multidim_sync"]
    scalar_async = out["arms"]["scalar_async"]

    sym_exact = bool(scalar_sync["frac_seeds_exactly_zero"] >= 1.0 - 1e-9)
    iso_exact = bool(iso_sync["frac_seeds_exactly_zero"] >= 1.0 - 1e-9)
    md_positive = bool(md_sync["rupture_asymmetry_mean"] > 0)
    sched_effect = out["contrasts"]["schedule_effect_scalar"]["mean_diff"]

    verdict = {}
    verdict["scalar_sync_is_exactly_symmetric"] = sym_exact
    verdict["isotropic_sync_is_exactly_symmetric"] = iso_exact
    verdict["multidim_sync_asymmetry_positive"] = md_positive
    verdict["async_scheduler_creates_asymmetry_in_scalar_model"] = bool(
        np.isfinite(sched_effect) and sched_effect > 0)
    if sym_exact and iso_exact and md_positive:
        verdict["conclusion"] = (
            "SUPPORTED_WITH_SCOPE: 同步调度下，标量模型与各向同性多维模型的断裂严格对称（A≡0），"
            "仅当引入『个体各向异性语义判据』后才产生非对称撕裂；同时异步调度本身即可在标量模型中"
            "制造非对称断裂，故原论文把 A_rupture 全部归因于『多维语义』属于过度归因，"
            "正确表述为『个体各向异性语义判据』与『异步调度』的联合效应。")
    else:
        verdict["conclusion"] = (
            "INCONCLUSIVE: 未能复现预期的严格对称性，请检查各臂判据实现。")
    out["verdict"] = verdict

    P("\n=== 归因判定 ===")
    for k, v in verdict.items():
        P(f"{k}: {v}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(a.out.replace(".json", ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\n[Done] Phase 4 attribution results written to {a.out}")


if __name__ == "__main__":
    main()
