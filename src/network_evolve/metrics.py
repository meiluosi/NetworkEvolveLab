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
