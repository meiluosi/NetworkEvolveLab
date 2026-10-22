"""实验工具：单配置运行、多种子并行、均值与 95% 置信区间、H1/H2 判定。"""
import numpy as np
from multiprocessing import Pool
from scipy import stats
from .engine import CoEvolutionEngine
from .metrics import (snapshot, init_incident_edges, retention_by_node, h2_test, h2_seed_level_test,
                      cross_reach)

BASE = dict(nodes=100, rounds=60, theta=0.4, bridger_dtheta=0.3, p_form=0.1,
            block=None, rho=1.0, hetero=True, checkpoints=(), reach=False)
ENGINE_KEYS = ("bridger_moderate", "bridger_stress", "bridger_links", "stress_decay", "stress_sens",
               "stress_scale", "stress_diff", "div_rec", "bridger_subsidy", "hub_limit", "theta_attract",
               "topology_type")


def make_engine(cfg, seed):
    c = {**BASE, **cfg}
    kw = {k: c[k] for k in ENGINE_KEYS if k in c}
    return CoEvolutionEngine(num_nodes=c["nodes"], seed=seed, base_tolerance=c["theta"],
                             bridger_tolerance=min(1.0, c["theta"] + c["bridger_dtheta"]),
                             p_form=c["p_form"], heterogeneous=c["hetero"], block_diff=c["block"],
                             rec_friction=c["rho"], **kw)


def run_config(args):
    cfg, seed = args
    c = {**BASE, **cfg}
    e = make_engine(cfg, seed)
    init = init_incident_edges(e)
    q0 = snapshot(e)
    traj = {}
    for r in range(1, c["rounds"] + 1):
        e.run_round()
        if r in c["checkpoints"]:
            traj[r] = snapshot(e)["modularity_louvain"]
    sn = snapshot(e)
    ret = retention_by_node(e, init)
    br = [v for i, v in ret.items() if e.agents[i].archetype == "active_bridger"]
    ot = [v for i, v in ret.items() if e.agents[i].archetype != "active_bridger"]
    # 同质回音室：规模>=10 且社群内信念方差<0.05 的 Louvain 社群数（H1 的“至少 2 个同质回音室”）
    import networkx as nx
    comms = nx.community.louvain_communities(e.graph.g.to_undirected(), weight="weight", seed=0)
    homo = sum(1 for cm in comms if len(cm) >= 10 and np.var([e.agents[i].belief for i in cm]) < 0.05)
    out = dict(q_init=q0["modularity_louvain"], q_end=sn["modularity_louvain"], q_sign=sn["modularity_q"],
               comms=sn["num_louvain_comms"], homo_comms=homo, polar=sn["polarization"],
               cross=sn["cross_edge_frac"], giant=sn["giant_frac"], isolated=sn["isolated_nodes"],
               big_comps=sn["num_big_components"], edges=sn["num_edges"], blocks=len(e.blocked),
               bridger_ret=br, other_ret=ot, traj=traj)
    if c["reach"]:
        out["reach"] = cross_reach(e, n_sims=150, seed=seed)
    return out


def run_many(cfg, seeds, procs=10):
    with Pool(procs) as p:
        return p.map(run_config, [(cfg, s) for s in seeds])


def mean_ci(xs, conf=0.95):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = stats.sem(xs) * stats.t.ppf((1 + conf) / 2, len(xs) - 1)
    return float(xs.mean()), float(h)


def fmt(xs):
    m, h = mean_ci(xs)
    return f"{m:.3f}±{h:.3f}"


def h2_summary(runs, n_boot=2000, seed=0):
    """合并所有种子的节点级留存率做 Mann-Whitney；并以“种子”为单位 bootstrap 相对差距 CI。

    统计口径说明（修复伪重复问题）：
      节点级 Mann-Whitney 把同一 seed 内的节点当作独立样本，会低估标准误（伪重复）。
      因此除节点级结果外，额外给出以「种子」为独立统计单元的判定：
      单样本单侧 t 检验 H0: mean(per-seed gap) <= 0.4  vs  H1: > 0.4，
      以及符号检验（达到 40% 差距的种子占比）。
      H2 的最终判定以 seed 级检验为准（seed_gap_p < 0.01）。
    """
    b = [v for r in runs for v in r["bridger_ret"]]
    o = [v for r in runs for v in r["other_ret"]]
    res = h2_test(b, o)
    gaps = []
    for r in runs:
        ob, oo = np.mean(r["bridger_ret"]), np.mean(r["other_ret"])
        gaps.append((oo - ob) / oo if oo > 0 else np.nan)
    gaps = np.array(gaps)
    rng = np.random.default_rng(seed)
    boots = [np.nanmean(rng.choice(gaps, size=len(gaps))) for _ in range(n_boot)]
    res["gap_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    seed_level = h2_seed_level_test(gaps, threshold=0.4)
    res["seed_level"] = seed_level
    res["p_value_node_level"] = res["p_value"]          # 保留原字段（伪重复，仅供参考）
    res["p_value"] = float(seed_level["p_value_seed_level"])   # 主判定口径改为 seed 级
    res["H2_supported"] = bool(seed_level["p_value_seed_level"] < 0.01
                                and seed_level["seed_gap_mean"] >= 0.4)
    return res


def h1_summary(runs):
    ok = [r["q_init"] < 0.25 and r["q_end"] > 0.65 and r["homo_comms"] >= 2 for r in runs]
    return dict(q_end=mean_ci([r["q_end"] for r in runs]), q_init=mean_ci([r["q_init"] for r in runs]),
                max_q_end=float(max(r["q_end"] for r in runs)), h1_pass_rate=float(np.mean(ok)),
                homo_comms=mean_ci([r["homo_comms"] for r in runs]))
