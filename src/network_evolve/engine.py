import argparse
import json
import numpy as np
from typing import Dict, List, Any
from .graph import DynamicSocialGraph
from .agent import EvolvingAgent

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
                 topology_type: str = "erdos_renyi"):
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
        self.graph = DynamicSocialGraph(num_nodes=num_nodes, seed=seed, topology_type=topology_type)
        
        # 初始化异质原型分布（延续 2% Hub, 8.5% Bridger, 89.5% Long-tail）
        self.agents: Dict[int, EvolvingAgent] = {}
        num_hubs = max(1, int(round(num_nodes * 0.02)))
        num_bridgers = max(1, int(round(num_nodes * 0.085)))
        
        # 初始信念设为两极化分布或均匀分布 [-1.0, 1.0]
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
        speaker_weights = []
        for a in self.agents.values():
            if not self.heterogeneous:
                w = 1.0
            elif a.archetype == "hub_broadcaster":
                w = 10.0
            elif a.archetype == "active_bridger":
                w = 3.0
            else:
                w = 0.5
            speaker_weights.append(w)
        speaker_weights = np.array(speaker_weights) / sum(speaker_weights)
        speaker_id = int(self.rng.choice(list(self.agents.keys()), p=speaker_weights))
        speaker = self.agents[speaker_id]
        
        # 2. 找到所有关注 speaker 的受众 (listener -> speaker)
        # 注意：在社交网络中，listener 关注 speaker，则边为 listener -> speaker
        listeners = list(self.graph.g.predecessors(speaker_id))
        if self.hub_limit > 0 and speaker.archetype == "hub_broadcaster":  # Hub 限流
            listeners = [u for u in listeners if self.rng.random() >= self.hub_limit]
                
        # 3. 每个受众做出反应，并调整边权重
        actions_taken = []
        for listener_id in listeners:
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
                self.graph.g.remove_edge(listener_id, speaker_id)
                self.blocked.add((listener_id, speaker_id))
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
        for _ in range(self.num_nodes):
            self.step(compute_metrics=False)
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
