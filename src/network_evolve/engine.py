import argparse
import json
import numpy as np
from typing import Any, Dict, List, Optional
from .graph import DynamicSocialGraph
from .agent import EvolvingAgent


class _BeliefArrayView:
    """把引擎的**标量**信念暴露成 1 维数组视图。

    统一效应层 ``llm_backend.apply_decision`` 以 ``world.b[i]``（向量）读写信念；
    旗舰引擎的状态是标量，因此用这层适配，而不是另写一份效应逻辑。
    """

    def __init__(self, agents: Dict[int, EvolvingAgent]):
        self.agents = agents

    def __getitem__(self, i):
        return np.array([float(self.agents[i].belief)], dtype=float)

    def __setitem__(self, i, value):
        self.agents[i].belief = float(np.asarray(value, dtype=float).ravel()[0])


class _EngineLLMAdapter:
    """让 CoEvolutionEngine 满足统一效应层所需的 ``world`` 接口。"""

    def __init__(self, engine: "CoEvolutionEngine"):
        self.engine = engine
        self.b = _BeliefArrayView(engine.agents)

    @property
    def g(self):
        return self.engine.graph.g

    def _drop_edge(self, u, sp):
        self.engine._drop_edge(u, sp)

class CoEvolutionEngine:
    """
    网络拓扑与智能体认知双向共演调度主引擎
    """
    def __init__(self, num_nodes: int = 50, seed: int = 42,
                 base_tolerance: float = 0.45, bridger_tolerance: float = 0.75,
                 p_form: float = 0.0, theta_attract: float = 0.4,
                 heterogeneous: bool = True, block_diff: float = None, rec_friction: float = 1.0,
                 bridger_moderate: bool = False, bridger_stress: bool = False,
                 bridger_links: int = 4, stress_decay: float = 0.9, stress_sens: float = 0.6,
                 stress_scale: float = 3.0, stress_diff: float = 0.5,
                 div_rec: float = 0.0, bridger_subsidy: float = 0.0, hub_limit: float = 0.0,
                 topology_type: str = "erdos_renyi", init_graph=None, init_beliefs=None,
                 llm_policy=None, llm_event_budget: int = 0,
                 initial_density: float = 0.08):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.p_form = p_form  # 三元闭包建边概率（0 表示关闭，保持原行为）
        self.theta_attract = theta_attract  # 信念距离小于该值才建边
        self.heterogeneous = heterogeneous  # False = Phase 1 对照组 A（同质行为，原型标签仅用于统计）
        self.block_diff = block_diff        # 信念差 >= 该值时拉黑（None 关闭）
        self.rec_friction = rec_friction    # 推荐摩擦力 ρ：以概率 ρ 同质性推荐，否则随机（多样性）推荐
        self.blocked = set()                # (listener, speaker) 拉黑对，禁止再次建边
        # —— Phase 1b：Bridger 受压机制（模型修正，需在预注册文档中记录）——
        self.bridger_stress = bridger_stress   # 动力学：累积来自左右两侧的失调暴露，压低有效容忍度
        self.stress_decay = stress_decay       # 每轮暴露记忆衰减
        self.stress_sens = stress_sens         # 压力对容忍度的最大折减比例
        self.stress_diff = stress_diff         # 信念差超过该值即计为一次失调暴露（无论是否触发衰减）
        self.stress_scale = stress_scale       # 暴露量饱和尺度
        self.exposure = {}                     # bridger_id -> [左侧暴露, 右侧暴露]
        # —— Phase 3：平台干预 ——
        self.div_rec = div_rec                 # 多样性推荐：新推荐中异质（对立）观点占比
        self.bridger_subsidy = bridger_subsidy # 对 Bridger 发言的衰减摩擦折减比例
        self.hub_limit = hub_limit             # Hub 发言触达限流比例
        # —— 统一 LLM 认知层（本次复核新增）——
        # 旗舰引擎原本只有**标量规则启发式**认知（EvolvingAgent.perceive_and_react），
        # 没有任何 LLM 参与；注入 llm_policy 后，引擎每步会有最多
        # llm_event_budget 个「听者-发言者」认知事件交由真实 LLM 决策，
        # 其余事件仍走规则后端（**混合式**，并在结果中显式报告 LLM 覆盖比例）。
        self.llm_policy = llm_policy
        self.llm_event_budget = int(llm_event_budget)
        # 每轮 LLM 事件总预算（引擎的一「轮」含 N 个单步，若不设上限，
        # LLM 调用数会被放大 N 倍；每轮预算才是可解释、可披露的旋钮）
        self._llm_round_left = self.llm_event_budget
        self._llm_plan: Dict[tuple, Any] = {}
        self._llm_plan_used: set = set()
        self._round_speakers: Optional[List[int]] = None
        self._step_index = 0
        #: 引擎自始至终发生的「听者-发言者」互动总数（LLM 覆盖率的真实分母）
        self.n_interactions = 0
        self.llm_stats = {"n_events": 0, "n_valid": 0, "n_invalid": 0, "n_calls": 0,
                           "n_cache_hits": 0, "n_rounds_with_llm": 0,
                           # n_applied: 被问到的事件中**真正落地**到拓扑的次数
                           # （有些计划事件因听众在轮内被拉黑/边先被切断而不会发生互动）
                           "n_applied": 0,
                           # actions: 已落地决策的动作分布
                           "actions": {}}
        self.graph = DynamicSocialGraph(num_nodes=num_nodes, seed=seed, topology_type=topology_type,
                                        init_graph=init_graph, initial_density=initial_density)

        # 初始化异质原型分布（延续 2% Hub, 8.5% Bridger, 89.5% Long-tail）
        self.agents: Dict[int, EvolvingAgent] = {}
        num_hubs = max(1, int(round(num_nodes * 0.02)))
        num_bridgers = max(1, int(round(num_nodes * 0.085)))

        # 初始信念设为两极化分布或均匀分布 [-1.0, 1.0]
        if init_beliefs is not None:
            # 导入外部（实证数据集）信念先验，保证与数据集初始化完全一致
            initial_beliefs = np.array([float(init_beliefs.get(i, 0.0)) for i in range(num_nodes)])
        else:
            initial_beliefs = self.rng.choice([-0.7, 0.7], size=num_nodes) + self.rng.normal(0, 0.15, size=num_nodes)
        initial_beliefs = np.clip(initial_beliefs, -1.0, 1.0)
        
        for i in range(num_nodes):
            if i < num_hubs:
                arch = "hub_broadcaster"
            elif i < num_hubs + num_bridgers:
                arch = "active_bridger"
            else:
                arch = "long_tail_responder"
            
            self.agents[i] = EvolvingAgent(
                agent_id=i,
                archetype=arch,
                belief=float(initial_beliefs[i]),
                tolerance=base_tolerance if (arch != "active_bridger" or not heterogeneous) else bridger_tolerance, # Bridger 具有更高的认知包容度
                damping=0.08
            )
            if arch == "active_bridger" and bridger_moderate:
                self.agents[i].belief = float(np.clip(self.rng.normal(0, 0.15), -1, 1))
            if arch == "active_bridger" and bridger_stress:
                self.exposure[i] = [0.0, 0.0]

        if bridger_moderate:  # Bridger 同时关注并被两侧关注
            ids = [i for i, a in self.agents.items() if a.archetype != "active_bridger"]
            left = [i for i in ids if self.agents[i].belief < 0]
            right = [i for i in ids if self.agents[i].belief >= 0]
            for b, a in self.agents.items():
                if a.archetype != "active_bridger":
                    continue
                for camp in (left, right):
                    if not camp:
                        continue
                    for k in self.rng.choice(camp, size=min(bridger_links, len(camp)), replace=False):
                        k = int(k)
                        self.graph.g.add_edge(b, k, weight=float(self.rng.uniform(0.4, 0.8)))
                        self.graph.g.add_edge(k, b, weight=float(self.rng.uniform(0.4, 0.8)))

    def step(self, compute_metrics: bool = True) -> Dict[str, Any]:
        """推进一个交互演化时间步"""
        # 1. 挑选发言者（Hub 发言概率显著高于 Long-tail）
        speaker_weights = self._speaker_weights()
        if getattr(self, "_round_speakers", None) is not None and self._step_index < len(self._round_speakers):
            speaker_id = self._round_speakers[self._step_index]
            self._step_index += 1
        else:
            agent_ids = getattr(self, "_agent_ids_cache", None)
            if agent_ids is None:
                agent_ids = list(self.agents.keys())
                self._agent_ids_cache = agent_ids
            speaker_id = int(self.rng.choice(agent_ids, p=speaker_weights))
        speaker = self.agents[speaker_id]

        # 2. 找到所有关注 speaker 的受众 (listener -> speaker)
        # 注意：在社交网络中，listener 关注 speaker，则边为 listener -> speaker
        listeners = list(self.graph.g.pred[speaker_id])
        if self.hub_limit > 0 and speaker.archetype == "hub_broadcaster":  # Hub 限流
            listeners = [u for u in listeners if self.rng.random() >= self.hub_limit]
                
        # 3. 每个受众做出反应，并调整边权重
        actions_taken = []
        # 记录本步的互动总数（LLM 覆盖率的真实分母；不触碰随机流）
        self.n_interactions += len(listeners)

        # 3a. 统一 LLM 认知层：本轮计划中命中的二元组采用真实 LLM 的决策
        #     （闭环：LLM 的取关/静音决策真实改变拓扑）；其余互动走规则后端。
        for listener_id in listeners:
            llm_res = None
            if self.llm_policy is not None and self._llm_plan:
                llm_res = self._llm_apply(listener_id, speaker_id)
            if llm_res is not None:
                actions_taken.append(llm_res)
                continue
            listener = self.agents[listener_id]
            diff_before = abs(listener.belief - speaker.belief)
            stressed = self.bridger_stress and listener_id in self.exposure
            if stressed:
                base_tol = listener.tolerance
                listener.tolerance = base_tol * (1.0 - self.stress_sens * self._stress(listener_id))
            res = listener.perceive_and_react(speaker.belief, speaker.archetype)
            if stressed:
                listener.tolerance = base_tol
                if diff_before > self.stress_diff:  # 记录失调暴露（左/右）
                    self.exposure[listener_id][0 if speaker.belief < listener.belief else 1] += 1.0
            if (self.bridger_subsidy > 0 and speaker.archetype == "active_bridger"
                    and res["weight_delta"] < 0):
                res["weight_delta"] *= (1.0 - self.bridger_subsidy)
            # 拓扑更新
            if (self.block_diff is not None and res["action"] == "decay"
                    and diff_before >= self.block_diff):
                self._drop_edge(listener_id, speaker_id)
                res["action"] = "block"
                new_w = 0.0
            else:
                new_w = self.graph.update_edge_weight(listener_id, speaker_id, res["weight_delta"])
            res["remaining_weight"] = new_w
            actions_taken.append(res)
            
        if not compute_metrics:
            return {"speaker_id": speaker_id, "num_listeners": len(listeners)}

        # 4. 计算当前全局指标
        beliefs = [a.belief for a in self.agents.values()]
        polarization = float(np.var(beliefs))
        # 划分社群（以信念正负粗略二分）
        partitions = {i: 0 if a.belief < 0 else 1 for i, a in self.agents.items()}
        modularity = self.graph.calculate_modularity(partitions)
        graph_stats = self.graph.get_summary_stats()
        
        return {
            "speaker_id": speaker_id,
            "speaker_archetype": speaker.archetype,
            "speaker_belief": speaker.belief,
            "num_listeners": len(listeners),
            "polarization_var": polarization,
            "modularity_q": modularity,
            "num_edges": graph_stats["num_edges"],
            "density": graph_stats["density"],
            "num_components": graph_stats["num_connected_components"]
        }

    def _speaker_weights(self) -> np.ndarray:
        """发言者抽样权重（按原型：Hub 10 / Bridger 3 / Long-tail 0.5）。

        性能注记：原型在初始化后**永不改变**，因此该权重向量是常数。
        原实现每个 step 重建一次 N 元素数组，在 $N=1000$ 时占单轮耗时的
        约 30%。这里改为惰性缓存，**数值逐位不变**（回归测试
        ``tests/test_optimizer_equivalence.py`` 对照优化前快照逐位核验）。
        """
        cached = getattr(self, "_spk_w_cache", None)
        if cached is not None:
            return cached
        w = []
        for a in self.agents.values():
            if not self.heterogeneous:
                w.append(1.0)
            elif a.archetype == "hub_broadcaster":
                w.append(10.0)
            elif a.archetype == "active_bridger":
                w.append(3.0)
            else:
                w.append(0.5)
        arr = np.array(w, dtype=float)
        arr = arr / arr.sum()
        self._spk_w_cache = arr
        return arr

    def _plan_llm_round(self, speakers: List[int]) -> None:
        """为本轮**一次性**规划 LLM 认知事件（单次批量调用）。

        为什么是「每轮一次」而不是「每步一次」：
        引擎的一轮包含 N 个单步，若每步都问一次 LLM，调用数会被放大 N 倍
        （实测 15 s/次），既昂贵又让 LLM 覆盖比例失去可解释性。
        这里在轮首按发言者分布采样 K 个「听者-发言者」二元组，
        **一次批量调用**取得这 K 个事件的决策；随后这些二元组在轮内真正互动时
        直接采用 LLM 决策（闭环：LLM 的取关/静音决策真实改变拓扑），
        其余互动仍走规则后端。

        预抽发言者序列不改变随机流：逐个抽取 N 次与一次性抽取 N 次得到同一序列。
        """
        from .llm_backend import (LLMEvent, message_text, verbalize)
        pol = self.llm_policy
        arch_map = {"hub_broadcaster": "hub", "active_bridger": "bridger",
                    "long_tail_responder": "tail"}
        events: List[LLMEvent] = []
        pairs: List[tuple] = []
        seen = set()
        budget = self.llm_event_budget
        # 先把本轮每个发言者的听众整理成候选池（按发言顺序）
        candidates = []
        for sp in speakers:
            for u in self.graph.g.predecessors(sp):
                if (u, sp) not in self.blocked and (u, sp) not in seen:
                    candidates.append((u, sp))
        if not candidates:
            self._llm_plan = {}
            return
        k = min(budget, len(candidates))
        idx = sorted(int(i) for i in self.rng.choice(len(candidates), size=k, replace=False))
        for i in idx:
            u, sp = candidates[i]
            seen.add((u, sp))
            lu = self.agents[u]
            sp_agent = self.agents[sp]
            events.append(LLMEvent(
                event_id=f"e{u}_s{sp}",
                listener=int(u), speaker=int(sp),
                listener_archetype=arch_map.get(lu.archetype, "tail"),
                listener_belief=[float(lu.belief)],
                listener_profile=verbalize([float(lu.belief)], pol.topics, pol.poles),
                edge_weight=float(self.graph.g[u][sp]["weight"]),
                message=message_text(arch_map.get(sp_agent.archetype, "tail"),
                                     [float(sp_agent.belief)], pol.topics, pol.poles),
                tolerance=float(lu.tolerance),
            ))
            pairs.append((u, sp))

        decisions, stats = pol.decide(events)
        self.llm_stats["n_events"] += stats["n_events"]
        self.llm_stats["n_valid"] += stats["n_valid"]
        self.llm_stats["n_invalid"] += stats["n_invalid"]
        if not stats["receipt"]["cached"]:
            self.llm_stats["n_calls"] += 1
        else:
            self.llm_stats["n_cache_hits"] += 1
        self._llm_plan = {pr: dec for pr, dec in zip(pairs, decisions)}

    def _llm_apply(self, listener_id: int, speaker_id: int) -> Optional[Dict[str, Any]]:
        """若本轮计划中包含该二元组且尚未使用，则采用 LLM 决策并落地。"""
        from .llm_backend import apply_decision
        key = (listener_id, speaker_id)
        dec = self._llm_plan.get(key)
        if dec is None or key in self._llm_plan_used:
            return None
        self._llm_plan_used.add(key)
        before = float(self.agents[listener_id].belief)
        adapter = _EngineLLMAdapter(self)
        out = apply_decision(adapter, listener_id, speaker_id, dec,
                             eta=self.agents[listener_id].damping)
        after = float(self.agents[listener_id].belief)
        self.llm_stats["n_applied"] += 1
        acts = self.llm_stats.setdefault("actions", {})
        acts[dec.action] = acts.get(dec.action, 0) + 1
        action = "block" if out["dropped"] else (
            "decay" if (out.get("weight_delta") or 0) < 0 else "reinforce")
        remaining = 0.0
        if not out["dropped"] and self.graph.g.has_edge(listener_id, speaker_id):
            remaining = float(self.graph.g[listener_id][speaker_id]["weight"])
        return {
            "agent_id": listener_id,
            "speaker_belief": float(self.agents[speaker_id].belief),
            "new_belief": after,
            "belief_shift": after - before,
            "agreement": 1.0 - min(1.0, abs(before - float(self.agents[speaker_id].belief)) / 2.0),
            "action": action,
            "llm_action": dec.action,
            "weight_delta": out.get("weight_delta", 0.0),
            "remaining_weight": remaining,
            "source": "llm",
            "reasoning": dec.reasoning,
        }

    def _drop_edge(self, u: int, v: int) -> bool:
        """断边并写入黑名单（禁止被三元闭包重新连回）。

        与 VecWorld._drop_edge 语义一致 —— 统一的「断裂」定义是审稿人可核对的前提。
        """
        if self.graph.g.has_edge(u, v):
            self.graph.g.remove_edge(u, v)
            self.blocked.add((u, v))
            return True
        return False

    def _stress(self, i: int) -> float:
        """双向压力：左右暴露越均衡、总量越大，压力越高（范围 0~1）"""
        l, r = self.exposure[i]
        tot = l + r
        if tot <= 0:
            return 0.0
        balance = 2.0 * min(l, r) / tot
        return float(balance * np.tanh(tot / self.stress_scale))

    def triadic_closure(self) -> int:
        """同质性三元闭包：i 关注 j、j 关注 k，若信念足够接近则以 p_form 概率 i->k"""
        if self.p_form <= 0:
            return 0
        added = 0
        g = self.graph.g
        for i in list(self.agents):
            if self.rng.random() >= self.p_form:
                continue
            if self.div_rec > 0 and self.rng.random() < self.div_rec:  # 多样性推荐：对立观点
                bi = self.agents[i].belief
                cands = [k for k in self.agents if k != i and not g.has_edge(i, k)
                         and (self.agents[k].belief < 0) != (bi < 0)]
            elif self.rec_friction >= 1.0 or self.rng.random() < self.rec_friction:
                cands = {k for j in g.successors(i) for k in g.successors(j)} - set(g.successors(i)) - {i}
                cands = [k for k in cands
                         if abs(self.agents[i].belief - self.agents[k].belief) < self.theta_attract]
            else:  # 多样性推荐：忽略信念，从全体随机抽取
                cands = [k for k in self.agents if k != i and not g.has_edge(i, k)]
            cands = [k for k in cands if (i, k) not in self.blocked]
            if cands:
                k = int(self.rng.choice(cands))
                g.add_edge(i, k, weight=0.3)
                added += 1
        return added

    def run_round(self) -> Dict[str, Any]:
        """一轮 = N 次发言（单次 step 的 N 倍）+ 一次三元闭包"""
        if self.llm_policy is not None and self.llm_event_budget > 0:
            # 预抽本轮发言序列（与逐步抽取得到同一序列，不改变随机流）
            self._round_speakers = [int(self.rng.choice(list(self.agents.keys()),
                                                         p=self._speaker_weights()))
                                    for _ in range(self.num_nodes)]
            self._step_index = 0
            self._llm_plan_used = set()
            self._plan_llm_round(self._round_speakers)
            self.llm_stats["n_rounds_with_llm"] += 1
        for _ in range(self.num_nodes):
            self.step(compute_metrics=False)
        self._round_speakers = None
        for e in self.exposure.values():
            e[0] *= self.stress_decay
            e[1] *= self.stress_decay
        res = {}
        res["edges_added"] = self.triadic_closure()
        return res

    def run_simulation(self, steps: int = 30) -> List[Dict[str, Any]]:
        trajectory = []
        for s in range(steps):
            res = self.step()
            res["step"] = s
            trajectory.append(res)
        return trajectory

def main():
    parser = argparse.ArgumentParser(description="Run Network Co-Evolution Simulation")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--out", type=str, default="results/baseline_evolve.json")
    args = parser.parse_args()

    engine = CoEvolutionEngine(num_nodes=args.nodes)
    traj = engine.run_simulation(steps=args.steps)
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"nodes": args.nodes, "steps": args.steps}, "trajectory": traj}, f, indent=2)
    print(f"Simulation completed. Output written to {args.out}")

if __name__ == "__main__":
    main()
