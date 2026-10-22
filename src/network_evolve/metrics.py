import networkx as nx
import numpy as np
from typing import Dict
from scipy import stats


def snapshot(engine) -> Dict[str, float]:
    """Phase 0 宏观指标：Q、Π、最大弱连通分量占比、跨阵营边占比、Bridger 出边留存"""
    g = engine.graph.g
    agents = engine.agents
    beliefs = np.array([a.belief for a in agents.values()])
    part = {i: int(a.belief >= 0) for i, a in agents.items()}
    m = g.number_of_edges()
    cross = sum(part[u] != part[v] for u, v in g.edges())
    giant = max((len(c) for c in nx.weakly_connected_components(g)), default=0)
    und = g.to_undirected()
    comms = nx.community.louvain_communities(und, weight="weight", seed=0) if m else [set(g)]
    comp_sizes = [len(c) for c in nx.weakly_connected_components(g)]
    # 回音室同质性：社群内信念方差（越小越同质）
    within = float(np.mean([np.var([agents[i].belief for i in c]) for c in comms if len(c) > 1] or [0.0]))
    return {
        "modularity_louvain": float(nx.community.modularity(und, comms, weight="weight")) if m else 0.0,
        "num_louvain_comms": len(comms),
        "within_comm_belief_var": within,
        "isolated_nodes": sum(1 for _, d in g.degree() if d == 0),
        "num_big_components": sum(1 for x in comp_sizes if x >= 3),
        "modularity_q": engine.graph.calculate_modularity(part),
        "polarization": float(np.var(beliefs)),
        "num_edges": m,
        "cross_edge_frac": cross / m if m else 0.0,
        "giant_frac": giant / g.number_of_nodes(),
        "num_components": nx.number_weakly_connected_components(g),
        "small_comm_frac": float(np.mean([len(c) <= 5 for c in comms])) if comms else 0.0,
    }


def engine_metrics(engine, init_edges) -> Dict[str, float]:
    """把旗舰引擎 CoEvolutionEngine 的终态折算为与 baselines 同口径的基准指标。

    口径与 baselines.BaseOpinionModel.get_metrics 完全一致（Louvain 模块度、信念方差极化度、
    最大弱连通巨片占比、小社群占比、互惠二元组非对称断裂率），确保横向对比是同尺度的。
    """
    g = engine.graph.g
    und = g.to_undirected()
    m = g.number_of_edges()
    if m > 0:
        comms = nx.community.louvain_communities(und, weight="weight", seed=0)
        q = float(nx.community.modularity(und, comms, weight="weight"))
    else:
        comms, q = [set(engine.agents.keys())], 0.0
    beliefs = np.array([a.belief for a in engine.agents.values()], dtype=float)
    sizes = [len(c) for c in comms]
    giant = max((len(c) for c in nx.weakly_connected_components(g)), default=0) / max(g.number_of_nodes(), 1)
    asym = asymmetry_stats(g, init_edges)
    return {
        "modularity_q": q,
        "polarization": float(np.var(beliefs)),
        "giant_frac": float(giant),
        "small_comm_frac": float(np.mean([s <= 5 for s in sizes])) if sizes else 0.0,
        "rupture_asymmetry": asym["rupture_asymmetry"],
        "num_edges": float(m),
        "num_comms": float(len(comms)),
        "n_mutual_broken": asym["n_mutual_broken"],
    }


def init_incident_edges(engine):
    """记录初始每个节点的关联边（入+出），用于计算连边留存率"""
    g = engine.graph.g
    return {i: set(g.in_edges(i)) | set(g.out_edges(i)) for i in engine.agents}


def retention_by_node(engine, init_edges) -> Dict[int, float]:
    g = engine.graph.g
    return {i: (sum(g.has_edge(*e) for e in es) / len(es)) for i, es in init_edges.items() if es}


def h2_test(bridger_ret, other_ret) -> Dict[str, float]:
    """H2：Bridger 留存率应比其余节点低 >=40% 且 p<0.01（单侧 Mann-Whitney）"""
    b, o = np.array(bridger_ret), np.array(other_ret)
    rel_gap = (o.mean() - b.mean()) / o.mean() if o.mean() > 0 else float("nan")
    p = stats.mannwhitneyu(b, o, alternative="less").pvalue
    return {"bridger_mean": float(b.mean()), "other_mean": float(o.mean()),
            "relative_gap": float(rel_gap), "p_value": float(p),
            "H2_supported": bool(rel_gap >= 0.4 and p < 0.01)}


def h2_seed_level_test(seed_gaps, threshold: float = 0.4) -> Dict[str, float]:
    """以「种子」为独立统计单元的 H2 判定（避免节点级伪重复）。

    seed_gaps: 每个种子的相对留存率差距 (other_mean - bridger_mean) / other_mean。
    H2 要求差距 >= threshold（即 Bridger 留存率至少低 40%）。
    预注册判据「至少低 40% 且 p<0.01」拆为两个独立部分：
      · 效应量：mean(seed_gap) >= threshold（0.40）
      · 显著性：单样本单侧 t 检验 H0: mean(gap) <= 0 vs H1: > 0
    另附更严格的对照检验 p_value_threshold（H0: mean(gap) <= 0.40），仅作参考。
    """
    g = np.asarray([x for x in seed_gaps if np.isfinite(x)], dtype=float)
    if len(g) < 2:
        return {"seed_gap_mean": float(g.mean()) if len(g) else float("nan"),
                "seed_gap_median": float(np.median(g)) if len(g) else float("nan"),
                "p_value_seed_level": float("nan"),
                "frac_seeds_passing": float("nan"), "n_seeds": int(len(g))}
    if np.allclose(g, g[0]):
        # 退化情形：所有种子的差距完全相同（方差为 0），t 检验无定义
        p_pos = 1.0 if g[0] <= 0 else 0.0
        p_thr = 1.0 if g[0] <= threshold else 0.0
    else:
        _, p_pos = stats.ttest_1samp(g, 0.0, alternative="greater")
        _, p_thr = stats.ttest_1samp(g, threshold, alternative="greater")
    # 符号检验：差距为正的种子比例（H0: p = 0.5）
    binom_p = float(stats.binomtest(int(np.sum(g > 0)), len(g), 0.5, alternative="greater").pvalue)
    frac = float(np.mean(g >= threshold))
    return {"seed_gap_mean": float(g.mean()), "seed_gap_median": float(np.median(g)),
            "p_value_seed_level": float(p_pos), "p_value_threshold": float(p_thr),
            "p_value_sign_test": binom_p,
            "frac_seeds_passing": frac, "n_seeds": int(len(g))}


def asymmetry_stats(g, initial_edges) -> Dict[str, float]:
    """互惠二元组中的非对称断裂统计。

    仅统计初始为互惠 (u<->v) 的二元组：任一侧断裂记为 any_broken，
    其中仅一侧断裂（另一侧保留）记为 one_sided。非对称断裂率 = one_sided / any_broken。
    当没有任何互惠二元组断裂时返回 NaN，同时给出分母，避免 NaN 被误读为 0。
    """
    mutual = {(u, v) for (u, v) in initial_edges if (v, u) in initial_edges and u < v}
    one_broken = 0
    any_broken = 0
    for u, v in mutual:
        has_uv, has_vu = g.has_edge(u, v), g.has_edge(v, u)
        if not (has_uv and has_vu):
            any_broken += 1
            if has_uv != has_vu:
                one_broken += 1
    asym = one_broken / any_broken if any_broken else float("nan")
    return {"rupture_asymmetry": float(asym), "n_mutual": float(len(mutual)),
            "n_mutual_broken": float(any_broken), "n_one_sided": float(one_broken)}


def cross_reach(engine, n_sims: int = 200, beta: float = 0.5, seed: int = 123) -> float:
    """跨群有效信息到达率：独立级联，信息沿“关注”反方向流动（speaker -> follower）。
    每条边以 beta*weight 概率传递；从随机源出发，统计对立阵营（按信念正负）被触达的比例，取平均。"""
    rng = np.random.default_rng(seed)
    g = engine.graph.g
    side = {i: int(a.belief >= 0) for i, a in engine.agents.items()}
    nodes = list(g.nodes)
    vals = []
    for _ in range(n_sims):
        src = int(rng.choice(nodes))
        opp = [i for i in nodes if side[i] != side[src]]
        if not opp:
            continue
        seen, frontier = {src}, [src]
        while frontier:
            nxt = []
            for v in frontier:
                for u in g.predecessors(v):
                    if u not in seen and rng.random() < beta * g[u][v]["weight"]:
                        seen.add(u)
                        nxt.append(u)
            frontier = nxt
        vals.append(sum(1 for i in seen if side[i] != side[src]) / len(opp))
    return float(np.mean(vals)) if vals else 0.0
