"""Phase 2：多维语义信念 + 可插拔认知后端 + 成本记账。

后端:
  - degroot : DeGroot 加权平均（不取关）
  - hk      : Hegselmann-Krause 有界信任（超出 epsilon 即取关）
  - proxy   : 离线"类 LLM"规则代理（显著维度加权、逆火、极端语气惩罚）——只是对 LLM 行为的假设，不是 LLM
  - llm     : 真实 LLM 后端。**本次复核新增**（此前本文件头声称有该后端，但 ``react()``
              里并没有 llm 分支，真实 LLM 通路是一条旁路）。
              现在由 ``llm_backend.LLMCognitivePolicy`` 统一实现决策，
              效应层与其它后端共用 ``llm_backend.apply_decision``；
              决策可离线重放（内容寻址归档），并记录完整收据
              （模型标识 / temperature / token 计数）。
"""
import json
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List
from .metrics import asymmetry_stats


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
                 theta_attract=0.5, damping=0.1, update_mode="async",
                 isotropic=False, symmetric_tol=False, proxy_tol=0.35,
                 uniform_speaker=False,
                 b_init=None, g0=None, salience_init=None, llm_policy=None):
        self.rng = np.random.default_rng(seed)
        self.n, self.d, self.backend, self.eps = n, d, backend, eps
        self.p_form, self.theta_attract, self.damping = p_form, theta_attract, damping
        # —— 归因消融开关（Phase 4）——
        # update_mode: "async" = 每步随机单一发言者（原始实现）；"sync" = 每轮全体同步更新
        # isotropic   : True 时全体智能体使用统一议题权重（各向同性），用于剥离“个体各向异性”这一变量
        # symmetric_tol: True 时容忍度只依赖听者（不随发言者语气激进度变化），保证判据严格对称
        # uniform_speaker: True 时全体发言概率均匀（1/n），剥离原型发言倾向偏置
        # b_init / g0 / salience_init: 允许不同实验臂共享完全相同的初始条件（受控对比）
        self.update_mode = update_mode
        self.isotropic = isotropic
        self.symmetric_tol = symmetric_tol
        self.proxy_tol = proxy_tol
        self.uniform_speaker = uniform_speaker
        if b_init is not None:
            # 受控初始条件：直接采用外部给定的信念矩阵（形状须为 (n, d)）
            b = np.array(b_init, dtype=float)
            if b.shape != (n, d):
                raise ValueError(f"b_init shape {b.shape} != ({n}, {d})")
            self.b = np.clip(b, -1, 1)
        else:
            camp = self.rng.choice([-1.0, 1.0], size=n)
            b = self.rng.normal(0, 0.25, size=(n, d))
            b[:, 0] += 0.6 * camp                         # 核心议题维度：两阵营
            # 次要维度与阵营弱相关（交叉议题）：部分人在次要维度上跨阵营
            b[:, 1:] += 0.3 * camp[:, None] * self.rng.choice([-1.0, 1.0], size=(n, d - 1))
            self.b = np.clip(b, -1, 1)
        if salience_init is not None:
            self.salience = np.array(salience_init, dtype=float)
        elif isotropic:
            # 各向同性：所有人对各议题等权（1/d），个体间不存在偏好差异
            self.salience = np.full((n, d), 1.0 / d)
        else:
            self.salience = self.rng.dirichlet(np.ones(d) * 0.7, size=n)   # 每人对各议题的重视度
        self.hub = set(range(max(1, round(n * 0.02))))
        self.bridger = set(range(len(self.hub), len(self.hub) + max(1, round(n * 0.085))))
        if g0 is None:
            g0 = nx.erdos_renyi_graph(n, p_edge, directed=True, seed=seed)
        self.g = nx.DiGraph()
        self.g.add_nodes_from(range(n))
        for u, v in g0.edges():
            w = g0[u][v].get("weight", None)
            if w is None:
                w = float(self.rng.uniform(0.3, 0.9))
            self.g.add_edge(u, v, weight=float(w))
        self.initial_edges = set(self.g.edges())
        self.broken_edges = []
        # 断边黑名单：与 CoEvolutionEngine.blocked 语义一致 —— 一旦断连，不再被三元闭包重新连回。
        # 若缺失该约束，closure 会把刚断裂的边立即加回，使「断裂」在终态上被抹掉，
        # 并在互惠二元组上制造虚假的单向断裂（实测：isotropic 对照臂残留 A>0 全部源于此）。
        self.blocked = set()
        if uniform_speaker:
            self.speak_p = np.full(n, 1.0 / n)
        else:
            w = np.array([10.0 if i in self.hub else 3.0 if i in self.bridger else 0.5 for i in range(n)])
            self.speak_p = w / w.sum()
        self.ledger = None
        # ---- 统一 LLM 认知后端（本次复核新增；此前文件头声称有 llm 后端但 react() 无此分支）----
        # llm_policy 为 None 时 backend="llm" 不可用；注入后 LLM 决策与 proxy/hk/degroot
        # 走**完全相同**的效应层（llm_backend.apply_decision），杜绝语义分叉。
        self.llm_policy = llm_policy
        if self.backend == "llm" and llm_policy is None:
            raise ValueError(
                "backend='llm' 需要注入 llm_policy（见 src/network_evolve/llm_backend.py）。")
        # 每轮 LLM 事件总预算（VecWorld 的一「轮」含 N 个单步，不设上限会把调用数放大 N 倍）
        self.llm_round_budget = int(getattr(llm_policy, "round_budget", 0) or 0) if llm_policy else 0
        self._llm_round_left = self.llm_round_budget
        self.llm_stats = {"n_events": 0, "n_valid": 0, "n_invalid": 0, "n_calls": 0,
                           "n_cache_hits": 0, "n_rounds_with_llm": 0}

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
            # symmetric_tol=True：容忍度只依赖听者自身，使断裂判据在个体间严格对称（归因对照用）
            tol = self.proxy_tol if self.symmetric_tol else (0.35 - 0.15 * extremity)
            if sal_dist <= tol:
                return self.damping * diff * align_core, 0.05, "reinforce"
            if sal_dist > 2.2 * tol:                                # 逆火：被激怒后向反方向微移
                return -0.3 * self.damping * diff, -0.25, "backfire"
            return np.zeros(self.d), -0.12, "decay"
        raise ValueError(self.backend)

    def step(self):
        """推进一个时间步。update_mode 决定调度：async=单发言者事件，sync=全体同步更新。"""
        if self.update_mode == "sync":
            return self._step_sync()
        if self.backend == "llm":
            return self._step_llm()
        sp = int(self.rng.choice(self.n, p=self.speak_p))
        for u in list(self.g.predecessors(sp)):
            shift, dw, action = self.react(u, sp)
            self.b[u] = np.clip(self.b[u] + shift, -1, 1)
            if dw != 0.0:
                w = self.g[u][sp]["weight"] + dw
                if w <= 0:
                    self._drop_edge(u, sp)
                else:
                    self.g[u][sp]["weight"] = min(1.0, w)

    # ---- 统一 LLM 后端路径 ----
    def _llm_event(self, u, sp, tag=""):
        """把一个 (listener, speaker) 二元组构造成 LLM 认知事件。"""
        from .llm_backend import LLMEvent, archetype_of, message_text, verbalize
        pol = self.llm_policy
        topics = pol.topics
        poles = pol.poles
        return LLMEvent(
            event_id=f"{tag}u{u}_s{sp}",
            listener=int(u), speaker=int(sp),
            listener_archetype=archetype_of(self.hub, self.bridger, int(u)),
            listener_belief=[float(x) for x in self.b[u]],
            listener_profile=verbalize(self.b[u], topics, poles),
            edge_weight=float(self.g[u][sp]["weight"]),
            message=message_text(archetype_of(self.hub, self.bridger, int(sp)),
                                 self.b[sp], topics, poles),
            tolerance=float(self.proxy_tol),
        )

    def _step_llm(self):
        """LLM 认知后端的一步：把该步的全部听者-发言者事件**批量**交给 LLM 决策。

        与 proxy / hk / degroot 后端的差异只在「决策从哪来」，
        效应层（信念阻尼更新、边权重增减、断边黑名单）完全共用
        ``llm_backend.apply_decision``，因此对比是公平的。
        """
        from .llm_backend import apply_decision
        if self._llm_round_left <= 0:
            return
        sp = int(self.rng.choice(self.n, p=self.speak_p))
        listeners = list(self.g.predecessors(sp))
        if not listeners:
            return
        k = min(len(listeners), self._llm_round_left)
        self._llm_round_left -= k
        if k < len(listeners):
            idx = sorted(int(i) for i in self.rng.choice(len(listeners), size=k, replace=False))
            listeners = [listeners[i] for i in idx]
        events = [self._llm_event(u, sp) for u in listeners]
        decisions, stats = self.llm_policy.decide(events)
        self.llm_stats["n_events"] += stats["n_events"]
        self.llm_stats["n_valid"] += stats["n_valid"]
        self.llm_stats["n_invalid"] += stats["n_invalid"]
        if not stats["receipt"]["cached"]:
            self.llm_stats["n_calls"] += 1
        else:
            self.llm_stats["n_cache_hits"] += 1
        for ev, dec in zip(events, decisions):
            apply_decision(self, ev.listener, ev.speaker, dec, eta=self.damping)

    def _step_sync(self):
        """同步更新：所有有向边 (u -> sp) 上的听者 u 同时对 sp 做出反应。

        所有反应量先在更新前的信念状态上计算完毕，再统一施加，避免遍历顺序引入的偏置。
        经典标量有界信任模型在同步调度下断裂判据完全对称（|b_i-b_j| 与阈值均相同），
        因此非对称断裂率应严格为 0；这是 Phase 4 归因消融的关键对照。
        """
        if self.backend == "llm":
            from .llm_backend import apply_decision
            events = [self._llm_event(u, sp, tag="sync_")
                      for u, sp in list(self.g.edges())]
            if not events:
                return
            decisions, stats = self.llm_policy.decide(events)
            self.llm_stats["n_events"] += stats["n_events"]
            self.llm_stats["n_valid"] += stats["n_valid"]
            self.llm_stats["n_invalid"] += stats["n_invalid"]
            if not stats["receipt"]["cached"]:
                self.llm_stats["n_calls"] += 1
            else:
                self.llm_stats["n_cache_hits"] += 1
            for ev, dec in zip(events, decisions):
                apply_decision(self, ev.listener, ev.speaker, dec, eta=self.damping)
            return
        pending = []
        for u, sp in list(self.g.edges()):
            shift, dw, action = self.react(u, sp)
            pending.append((u, sp, shift, dw))
        for u, sp, shift, dw in pending:
            self.b[u] = np.clip(self.b[u] + shift, -1, 1)
            if dw != 0.0 and self.g.has_edge(u, sp):
                w = self.g[u][sp]["weight"] + dw
                if w <= 0:
                    self._drop_edge(u, sp)
                else:
                    self.g[u][sp]["weight"] = min(1.0, w)

    def apply_event_pairs(self, pairs):
        """按给定的 (listener, speaker) 事件序列执行本后端的认知反应。

        用途：Phase 2 的 LLM 臂与离线臂（DeGroot / HK / Proxy）必须面对**完全相同的
        发言事件序列**，否则「LLM vs 标量模型」的对比会混入轮数/事件数差异。
        本方法让离线后端重放归档的事件序列，从而实现事件级对齐的比较。
        """
        applied = 0
        for u, sp in pairs:
            if not self.g.has_edge(u, sp):
                continue
            shift, dw, action = self.react(u, sp)
            self.b[u] = np.clip(self.b[u] + shift, -1, 1)
            applied += 1
            if dw != 0.0:
                w = self.g[u][sp]["weight"] + dw
                if w <= 0:
                    self._drop_edge(u, sp)
                else:
                    self.g[u][sp]["weight"] = min(1.0, w)
        return applied

    def _drop_edge(self, u, sp):
        """移除边并写入黑名单（禁止被三元闭包重新连回）。"""
        if self.g.has_edge(u, sp):
            self.g.remove_edge(u, sp)
            self.broken_edges.append((u, sp))
            self.blocked.add((u, sp))

    def closure(self):
        for i in range(self.n):
            if self.rng.random() >= self.p_form:
                continue
            fo = set(self.g.successors(i))
            c = {k for j in fo for k in self.g.successors(j)} - fo - {i}
            c = [k for k in c if np.linalg.norm(self.b[i] - self.b[k]) / np.sqrt(self.d) < self.theta_attract
                 and (i, k) not in self.blocked]
            if c:
                self.g.add_edge(i, int(self.rng.choice(c)), weight=0.3)

    def run(self, rounds=30):
        for _ in range(rounds):
            self._llm_round_left = self.llm_round_budget
            if self.llm_round_budget > 0 and self.llm_policy is not None:
                self.llm_stats["n_rounds_with_llm"] += 1
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
        asym_stats = asymmetry_stats(self.g, self.initial_edges)
        asym = asym_stats["rupture_asymmetry"]
        bel = self.b
        pol = float(np.mean(np.sum((bel - bel.mean(0)) ** 2, axis=1)))
        return dict(modularity=float(q), num_comms=len(comms), size_gini=gini, small_comm_frac=small,
                    rupture_asymmetry=asym, polarization=pol, edges=m, giant_frac=
                    max(len(c) for c in nx.weakly_connected_components(self.g)) / self.n,
                    n_mutual=asym_stats["n_mutual"], n_mutual_broken=asym_stats["n_mutual_broken"],
                    n_one_sided=asym_stats["n_one_sided"],
                    cross_camp_edge_frac=float(np.mean([(self.b[u, 0] > 0) != (self.b[v, 0] > 0)
                                                        for u, v in self.g.edges()])) if m else 0.0)
