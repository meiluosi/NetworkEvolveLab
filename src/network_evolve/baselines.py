"""对比基线与前沿 SOTA 方法集成套件（Baselines & SOTA Frameworks）：
包含 6 大经典与最新前沿观点动力学模型：
1. DeGroot (1974): 经典线性加权共识模型
2. Hegselmann-Krause (2002): 经典有界信任模型（标量刚性断连）
3. Friedkin-Johnsen (1990): 顽固节点信念锚定模型
4. Generative Agents (Park et al., 2023): 记忆流与反思架构（固定拓扑）
5. Prompt-Rewire (Gu et al., 2025): 纯自然语言 Prompt 启发式重连模型
6. Ours (NetworkEvolveLab): 异质原型阻尼微调 + 加权拉黑拓扑共演框架
"""
import copy
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional

class BaseOpinionModel:
    """观点与拓扑动力学抽象基类"""
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.initial_g = copy.deepcopy(g)
        self.g = copy.deepcopy(g)
        self.n = self.g.number_of_nodes()
        self.b = np.array([initial_beliefs[i] for i in range(self.n)], dtype=float)
        self.initial_b = self.b.copy()
        self.initial_edges = set(self.g.edges())

    def run(self, rounds: int = 30) -> Dict[str, Any]:
        for _ in range(rounds):
            self.step()
        return self.get_metrics()

    def get_metrics(self) -> Dict[str, float]:
        und = self.g.to_undirected()
        m = self.g.number_of_edges()
        if m > 0:
            comms = nx.community.louvain_communities(und, weight="weight", seed=0)
            q = float(nx.community.modularity(und, comms, weight="weight"))
        else:
            comms, q = [set(range(self.n))], 0.0

        # 非对称断裂率 (Asymmetric Rupture Rate)
        mutual = {(u, v) for (u, v) in self.initial_edges if (v, u) in self.initial_edges and u < v}
        one_broken, any_broken = 0, 0
        for u, v in mutual:
            has_uv, has_vu = self.g.has_edge(u, v), self.g.has_edge(v, u)
            if not (has_uv and has_vu):
                any_broken += 1
                if has_uv != has_vu:
                    one_broken += 1
        asym = float(one_broken / any_broken) if any_broken > 0 else float("nan")

        # 连通巨片与孤立社群
        sizes = [len(c) for c in comms]
        giant = max(sizes) / self.n if sizes else 1.0
        small_comm = float(np.mean([s <= 5 for s in sizes])) if sizes else 0.0

        # 观点极化指数
        pol = float(np.var(self.b))

        return {
            "modularity_q": q,
            "polarization": pol,
            "giant_frac": giant,
            "small_comm_frac": small_comm,
            "rupture_asymmetry": asym,
            "num_edges": float(m),
            "num_comms": float(len(comms))
        }

# -------------------------------------------------------------
# 1. DeGroot (1974) 加权平均共识模型
# -------------------------------------------------------------
class DeGrootModel(BaseOpinionModel):
    """拓扑静态固定，观点每轮按入边权重加权平均更新，必然坍缩为共识"""
    def step(self):
        new_b = self.b.copy()
        for i in range(self.n):
            preds = list(self.g.predecessors(i))
            if preds:
                weights = [self.g[j][i].get("weight", 1.0) for j in preds]
                tot = sum(weights)
                new_b[i] = sum(self.b[j] * w for j, w in zip(preds, weights)) / tot
        self.b = new_b

# -------------------------------------------------------------
# 2. Hegselmann-Krause (2002) 经典有界信任模型
# -------------------------------------------------------------
class HegselmannKrauseModel(BaseOpinionModel):
    """标量有界信任：分歧 > epsilon 时直接断开/忽略，必然产生对称撕裂"""
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], epsilon: float = 0.45, seed: int = 42):
        super().__init__(g, initial_beliefs, seed)
        self.epsilon = epsilon

    def step(self):
        new_b = self.b.copy()
        to_remove = []
        for i in range(self.n):
            preds = list(self.g.predecessors(i))
            trusted = [j for j in preds if abs(self.b[i] - self.b[j]) <= self.epsilon]
            untrusted = [j for j in preds if abs(self.b[i] - self.b[j]) > self.epsilon]
            # 超出信任域断开
            for j in untrusted:
                to_remove.append((j, i))
            if trusted:
                new_b[i] = float(np.mean([self.b[j] for j in trusted]))
        for u, v in to_remove:
            if self.g.has_edge(u, v):
                self.g.remove_edge(u, v)
        self.b = new_b

# -------------------------------------------------------------
# 3. Friedkin-Johnsen (1990) 顽固信念锚定模型
# -------------------------------------------------------------
class FriedkinJohnsenModel(BaseOpinionModel):
    """引入顽固系数 lambda_i，观点受邻居影响同时锚定初始信念 b_i(0)"""
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], stubbornness: float = 0.35, seed: int = 42):
        super().__init__(g, initial_beliefs, seed)
        self.lambdas = np.full(self.n, 1.0 - stubbornness)

    def step(self):
        new_b = self.b.copy()
        for i in range(self.n):
            preds = list(self.g.predecessors(i))
            lam = self.lambdas[i]
            if preds:
                social_mean = float(np.mean([self.b[j] for j in preds]))
                new_b[i] = lam * social_mean + (1.0 - lam) * self.initial_b[i]
        self.b = new_b

# -------------------------------------------------------------
# 4. Generative Agents (Park et al., 2023) 记忆流与反思模型
# -------------------------------------------------------------
class GenerativeAgentsMemoryModel(BaseOpinionModel):
    """
    Park et al. 范式：基于短时记忆流（Memory Stream）与重要性反思（Reflection）
    网络拓扑保持静态不变，观点更新具备多步历史反思粘性
    """
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], memory_window: int = 5, seed: int = 42):
        super().__init__(g, initial_beliefs, seed)
        self.memory_window = memory_window
        self.memories: Dict[int, List[float]] = {i: [self.b[i]] for i in range(self.n)}

    def step(self):
        new_b = self.b.copy()
        for i in range(self.n):
            preds = list(self.g.predecessors(i))
            if preds:
                # 检索过去记忆与当前观察
                spk = int(self.rng.choice(preds))
                msg_val = self.b[spk]
                self.memories[i].append(msg_val)
                if len(self.memories[i]) > self.memory_window:
                    self.memories[i].pop(0)
                # 反思加权：近期记忆与固有倾向融合
                recency_weights = np.linspace(0.5, 1.0, len(self.memories[i]))
                recency_weights /= recency_weights.sum()
                reflection_val = float(np.sum(np.array(self.memories[i]) * recency_weights))
                # 阻尼更新
                new_b[i] = float(np.clip(self.b[i] + 0.12 * (reflection_val - self.b[i]), -1.0, 1.0))
        self.b = new_b

# -------------------------------------------------------------
# 5. Prompt-Rewire (Gu et al., 2025) 纯自然语言 Prompt 启发式重连模型
# -------------------------------------------------------------
class GuEtAlPromptRewireModel(BaseOpinionModel):
    """
    Gu et al. (Uni Melbourne 2025) 范式：
    模拟无阻尼、离散单次自然语言决策的 Follow/Unfollow 启发式重连
    缺乏连续衰减阻尼与黑名单机制，易出现剧烈连边流失与振荡
    """
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], seed: int = 42):
        super().__init__(g, initial_beliefs, seed)

    def step(self):
        new_b = self.b.copy()
        # 挑选活跃发言者
        spk = int(self.rng.choice(self.n))
        spk_b = self.b[spk]
        
        for listener in list(self.g.predecessors(spk)):
            diff = abs(self.b[listener] - spk_b)
            # 模拟未校准的大模型单次极端判定（无连续阻尼）
            if diff > 0.8:  # 激怒直接断连
                self.g.remove_edge(listener, spk)
                # 负反馈激化
                new_b[listener] = float(np.clip(self.b[listener] - 0.05 * np.sign(spk_b), -1.0, 1.0))
            elif diff < 0.3:
                # 强化靠拢
                new_b[listener] = float(np.clip(self.b[listener] + 0.15 * (spk_b - self.b[listener]), -1.0, 1.0))
                
        # 启发式随机两跳建边
        for i in range(self.n):
            if self.rng.random() < 0.15:
                succ = list(self.g.successors(i))
                if succ:
                    two_hop = {k for j in succ for k in self.g.successors(j)} - set(succ) - {i}
                    if two_hop:
                        k = int(self.rng.choice(list(two_hop)))
                        self.g.add_edge(i, k, weight=0.5)
        self.b = new_b

# -------------------------------------------------------------
# 6. Ours: NetworkEvolveLab 异质原型阻尼共演模型
# -------------------------------------------------------------
class OursCoEvolutionModel(BaseOpinionModel):
    """
    本项目完整架构：
    - 异质行为原型 (Hub / Bridger / Long-tail)
    - 连续加权衰减 + 拉黑黑名单 (Blocklist)
    - 认知失调阻尼连续位移
    - 多样化推荐反事实治理接口
    """
    def __init__(self, g: nx.DiGraph, initial_beliefs: Dict[int, float], 
                 theta: float = 0.40, block_diff: float = 1.2, div_rec: float = 0.15, seed: int = 42):
        super().__init__(g, initial_beliefs, seed)
        self.theta = theta
        self.block_diff = block_diff
        self.div_rec = div_rec
        self.blocked = set()
        
        # 确立行为原型
        self.archetypes = {}
        n_hubs = max(1, int(round(self.n * 0.02)))
        n_bridgers = max(2, int(round(self.n * 0.085)))
        for i in range(self.n):
            if i < n_hubs:
                self.archetypes[i] = "hub"
            elif i < n_hubs + n_bridgers:
                self.archetypes[i] = "bridger"
            else:
                self.archetypes[i] = "tail"

    def step(self):
        # 挑选发言者（Hub 高声量）
        weights = [10.0 if self.archetypes[i] == "hub" else 3.0 if self.archetypes[i] == "bridger" else 0.5 
                   for i in range(self.n)]
        p = np.array(weights) / sum(weights)
        spk = int(self.rng.choice(self.n, p=p))
        spk_b = self.b[spk]
        
        for listener in list(self.g.predecessors(spk)):
            diff = abs(self.b[listener] - spk_b)
            tol = 0.70 if self.archetypes[listener] == "bridger" else self.theta
            if diff <= tol * 2.0:
                # 阻尼拉近
                shift = (spk_b - self.b[listener]) * 0.08 * (1.0 - diff / 2.0)
                self.b[listener] = float(np.clip(self.b[listener] + shift, -1.0, 1.0))
                # 边权重增强
                new_w = min(1.0, self.g[listener][spk].get("weight", 0.5) + 0.05)
                self.g[listener][spk]["weight"] = new_w
            else:
                # 认知失调衰减或拉黑
                if diff >= self.block_diff:
                    self.g.remove_edge(listener, spk)
                    self.blocked.add((listener, spk))
                else:
                    new_w = self.g[listener][spk].get("weight", 0.5) - 0.15
                    if new_w <= 0.0:
                        self.g.remove_edge(listener, spk)
                    else:
                        self.g[listener][spk]["weight"] = new_w

        # 同质闭包与去中心化推荐
        for i in range(self.n):
            if self.rng.random() < 0.10:
                if self.div_rec > 0 and self.rng.random() < self.div_rec:
                    # 跨群多样化推荐
                    opp = [k for k in range(self.n) if k != i and not self.g.has_edge(i, k) 
                           and (self.b[k] < 0) != (self.b[i] < 0) and (i, k) not in self.blocked]
                    if opp:
                        k = int(self.rng.choice(opp))
                        self.g.add_edge(i, k, weight=0.35)
                else:
                    # 同质两跳推荐
                    succ = list(self.g.successors(i))
                    two_hop = {k for j in succ for k in self.g.successors(j)} - set(succ) - {i}
                    cands = [k for k in two_hop if abs(self.b[i] - self.b[k]) < 0.4 and (i, k) not in self.blocked]
                    if cands:
                        k = int(self.rng.choice(cands))
                        self.g.add_edge(i, k, weight=0.35)
