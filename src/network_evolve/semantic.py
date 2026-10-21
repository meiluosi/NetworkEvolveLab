"""Phase 2：多维语义信念 + 可插拔认知后端 + 成本记账。

后端:
  - degroot : DeGroot 加权平均（不取关）
  - hk      : Hegselmann-Krause 有界信任（超出 epsilon 即取关）
  - proxy   : 离线“类 LLM”规则代理（显著维度加权、逆火、极端语气惩罚）——只是对 LLM 行为的假设，不是 LLM
  - llm     : 真实 LLM 后端（见 llm_backend.py），经 CostLedger 记账 + 缓存，默认预算 100 次调用
"""
import json
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CostLedger:
    """API 成本记账：调用次数上限 + 缓存命中统计"""
    budget_calls: int = 100
    calls: int = 0
    cache_hits: int = 0
    log: List[dict] = field(default_factory=list)

    def charge(self, tag: str, n_items: int = 1):
        if self.calls >= self.budget_calls:
            raise RuntimeError(f"API 预算耗尽: {self.calls}/{self.budget_calls}")
        self.calls += 1
        self.log.append({"call": self.calls, "tag": tag, "items": n_items})

    def to_dict(self):
        return {"budget_calls": self.budget_calls, "calls": self.calls,
                "cache_hits": self.cache_hits, "log": self.log}


class VecWorld:
    """N 个智能体，d 维信念；有向加权图 u->v 表示 u 关注 v"""

    def __init__(self, n=60, d=3, backend="hk", eps=0.7, seed=0, p_edge=0.1, p_form=0.1,
                 theta_attract=0.5, damping=0.1):
        self.rng = np.random.default_rng(seed)
        self.n, self.d, self.backend, self.eps = n, d, backend, eps
        self.p_form, self.theta_attract, self.damping = p_form, theta_attract, damping
        camp = self.rng.choice([-1.0, 1.0], size=n)
        b = self.rng.normal(0, 0.25, size=(n, d))
        b[:, 0] += 0.6 * camp                         # 核心议题维度：两阵营
        # 次要维度与阵营弱相关（交叉议题）：部分人在次要维度上跨阵营
        b[:, 1:] += 0.3 * camp[:, None] * self.rng.choice([-1.0, 1.0], size=(n, d - 1))
        self.b = np.clip(b, -1, 1)
        self.salience = self.rng.dirichlet(np.ones(d) * 0.7, size=n)   # 每人对各议题的重视度
        self.hub = set(range(max(1, round(n * 0.02))))
        self.bridger = set(range(len(self.hub), len(self.hub) + max(1, round(n * 0.085))))
        g0 = nx.erdos_renyi_graph(n, p_edge, directed=True, seed=seed)
        self.g = nx.DiGraph()
        self.g.add_nodes_from(range(n))
        for u, v in g0.edges():
            self.g.add_edge(u, v, weight=float(self.rng.uniform(0.3, 0.9)))
        self.initial_edges = set(self.g.edges())
        self.broken_edges = []
        w = np.array([10.0 if i in self.hub else 3.0 if i in self.bridger else 0.5 for i in range(n)])
        self.speak_p = w / w.sum()
        self.ledger = None

    # ---- 认知反应：返回 (belief_shift 向量, weight_delta, action) ----
    def react(self, i, j):
        bi, bj = self.b[i], self.b[j]
        diff = bj - bi
        if self.backend == "degroot":
            return self.damping * diff, 0.0, "keep"
        dist = np.linalg.norm(diff) / np.sqrt(self.d)
        if self.backend == "hk":
            if dist <= self.eps / 2:
                return self.damping * diff, 0.05, "reinforce"
            return np.zeros(self.d), -0.15, "decay"
        if self.backend == "proxy":
            s = self.salience[i]
            sal_dist = float(np.sum(s * np.abs(diff)))            # 显著议题加权分歧
            extremity = float(np.linalg.norm(bj) / np.sqrt(self.d))  # 语气激进度
            align_core = 1.0 - sal_dist / 2.0
            tol = 0.35 - 0.15 * extremity                           # 面对极端言论更不宽容
            if sal_dist <= tol:
                return self.damping * diff * align_core, 0.05, "reinforce"
            if sal_dist > 2.2 * tol:                                # 逆火：被激怒后向反方向微移
                return -0.3 * self.damping * diff, -0.25, "backfire"
            return np.zeros(self.d), -0.12, "decay"
        raise ValueError(self.backend)

    def step(self):
        sp = int(self.rng.choice(self.n, p=self.speak_p))
        for u in list(self.g.predecessors(sp)):
            shift, dw, action = self.react(u, sp)
            self.b[u] = np.clip(self.b[u] + shift, -1, 1)
            if dw != 0.0:
                w = self.g[u][sp]["weight"] + dw
                if w <= 0:
                    self.g.remove_edge(u, sp)
                    self.broken_edges.append((u, sp))
                else:
                    self.g[u][sp]["weight"] = min(1.0, w)

    def closure(self):
        for i in range(self.n):
            if self.rng.random() >= self.p_form:
                continue
            fo = set(self.g.successors(i))
            c = {k for j in fo for k in self.g.successors(j)} - fo - {i}
            c = [k for k in c if np.linalg.norm(self.b[i] - self.b[k]) / np.sqrt(self.d) < self.theta_attract]
            if c:
                self.g.add_edge(i, int(self.rng.choice(c)), weight=0.3)

    def run(self, rounds=30):
        for _ in range(rounds):
            for _ in range(self.n):
                self.step()
            self.closure()
        return self.metrics()

    def metrics(self):
        und = self.g.to_undirected()
        m = self.g.number_of_edges()
        if m:
            comms = nx.community.louvain_communities(und, weight="weight", seed=0)
            q = nx.community.modularity(und, comms, weight="weight")
        else:
            comms, q = [set(range(self.n))], 0.0
        sizes = sorted((len(c) for c in comms), reverse=True)
        # 长尾聚类：非平凡社群(>=3)的规模 Gini，及小社群(<=5)占比
        arr = np.array([x for x in sizes if x >= 3], dtype=float)
        gini = float(np.abs(arr[:, None] - arr[None, :]).sum() / (2 * len(arr) ** 2 * arr.mean())) if len(arr) else 0.0
        small = float(np.mean([x <= 5 for x in sizes]))
        # 非对称撕裂：初始互关(mutual)的二元组里，仅单向断开的比例（相对任一方断开者）
        mutual = {(u, v) for (u, v) in self.initial_edges if (v, u) in self.initial_edges and u < v}
        one, any_broken = 0, 0
        for u, v in mutual:
            a, b_ = self.g.has_edge(u, v), self.g.has_edge(v, u)
            if not (a and b_):
                any_broken += 1
                one += int(a != b_)
        asym = one / any_broken if any_broken else float("nan")
        bel = self.b
        pol = float(np.mean(np.sum((bel - bel.mean(0)) ** 2, axis=1)))
        return dict(modularity=float(q), num_comms=len(comms), size_gini=gini, small_comm_frac=small,
                    rupture_asymmetry=asym, polarization=pol, edges=m, giant_frac=
                    max(len(c) for c in nx.weakly_connected_components(self.g)) / self.n,
                    cross_camp_edge_frac=float(np.mean([(self.b[u, 0] > 0) != (self.b[v, 0] > 0)
                                                        for u, v in self.g.edges()])) if m else 0.0)
