import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set

class DynamicSocialGraph:
    """
    可演化的动态社交网络拓扑引擎
    支持有向加权边、增删节点、边衰减、同质性三元闭包形成
    支持多种初始网络拓扑生成（Erdos-Renyi、Barabasi-Albert无标度、Watts-Strogatz小世界、实证双极图）
    """
    def __init__(self, num_nodes: int = 50, initial_density: float = 0.08, seed: int = 42,
                 topology_type: str = "erdos_renyi", init_graph=None):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.topology_type = topology_type

        self.g = nx.DiGraph()
        self.g.add_nodes_from(range(num_nodes))

        if init_graph is not None:
            # 导入外部（实证数据集）拓扑：保留其自身边权，缺失时用固定默认权重
            for u, v in init_graph.edges():
                if u < num_nodes and v < num_nodes and u != v:
                    w = init_graph[u][v].get("weight", None)
                    self.g.add_edge(u, v, weight=float(w) if w is not None else 0.6)
            return

        if topology_type == "scale_free" or topology_type == "barabasi_albert":
            # Barabasi-Albert 无标度网络，体现幂律度分布
            m_param = max(2, int(round(num_nodes * initial_density / 2)))
            undirected_ba = nx.barabasi_albert_graph(num_nodes, m=m_param, seed=seed)
            # 赋予有向性（按双向或根据度赋予主导方向）
            for u, v in undirected_ba.edges():
                self.g.add_edge(u, v, weight=float(self.rng.uniform(0.3, 0.9)))
                if self.rng.random() < 0.6:  # 60% 互相关注
                    self.g.add_edge(v, u, weight=float(self.rng.uniform(0.3, 0.9)))
        elif topology_type == "small_world" or topology_type == "watts_strogatz":
            # Watts-Strogatz 小世界网络，高集聚系数与短平均路径
            k_param = max(4, int(round(num_nodes * initial_density)))
            if k_param % 2 != 0:
                k_param += 1
            k_param = min(k_param, num_nodes - 1)
            undirected_ws = nx.connected_watts_strogatz_graph(num_nodes, k=k_param, p=0.15, seed=seed)
            for u, v in undirected_ws.edges():
                self.g.add_edge(u, v, weight=float(self.rng.uniform(0.3, 0.9)))
                if self.rng.random() < 0.6:
                    self.g.add_edge(v, u, weight=float(self.rng.uniform(0.3, 0.9)))
        elif topology_type == "empirical_bipolar":
            # 实证双极社交子图生成（模拟真实Twitter/Higgs极化网络结构）：
            # 两个核心阵营具有密集的无标度连接，两群之间存在低密度跨界连接
            n_half = num_nodes // 2
            g1 = nx.barabasi_albert_graph(n_half, m=3, seed=seed)
            g2 = nx.barabasi_albert_graph(num_nodes - n_half, m=3, seed=seed + 100)
            for u, v in g1.edges():
                self.g.add_edge(u, v, weight=float(self.rng.uniform(0.4, 0.9)))
                self.g.add_edge(v, u, weight=float(self.rng.uniform(0.4, 0.9)))
            for u, v in g2.edges():
                u_off, v_off = u + n_half, v + n_half
                self.g.add_edge(u_off, v_off, weight=float(self.rng.uniform(0.4, 0.9)))
                self.g.add_edge(v_off, u_off, weight=float(self.rng.uniform(0.4, 0.9)))
            # 跨阵营弱连边
            n_cross = max(5, int(num_nodes * 0.15))
            for _ in range(n_cross):
                u = int(self.rng.choice(range(n_half)))
                v = int(self.rng.choice(range(n_half, num_nodes)))
                self.g.add_edge(u, v, weight=float(self.rng.uniform(0.2, 0.5)))
                self.g.add_edge(v, u, weight=float(self.rng.uniform(0.2, 0.5)))
        else:
            # 默认：Erdos-Renyi 随机图
            base_g = nx.erdos_renyi_graph(num_nodes, p=initial_density, directed=True, seed=seed)
            for u, v in base_g.edges():
                if u != v:
                    self.g.add_edge(u, v, weight=float(self.rng.uniform(0.3, 0.9)))

    def get_neighbors(self, node_id: int) -> List[int]:
        """获取 node_id 关注的目标节点 (node_id -> v)"""
        return list(self.g.successors(node_id))

    def update_edge_weight(self, u: int, v: int, delta: float) -> float:
        """更新 edge (u -> v) 的权重，若 <= 0 则删除连边 (Unfollow)"""
        if self.g.has_edge(u, v):
            new_w = self.g[u][v]["weight"] + delta
            if new_w <= 0.0:
                self.g.remove_edge(u, v)
                return 0.0
            else:
                new_w = min(1.0, new_w)
                self.g[u][v]["weight"] = new_w
                return new_w
        elif delta > 0.0:
            # 建立新连边
            self.g.add_edge(u, v, weight=min(1.0, delta))
            return min(1.0, delta)
        return 0.0

    def calculate_modularity(self, partitions: Dict[int, int]) -> float:
        """
        计算给定社群划分下的无向化模块度 Q
        """
        undirected = self.g.to_undirected()
        if undirected.number_of_edges() == 0:
            return 0.0
        communities = {}
        for node, comm in partitions.items():
            communities.setdefault(comm, set()).add(node)
        comm_list = list(communities.values())
        try:
            return float(nx.community.modularity(undirected, comm_list))
        except Exception:
            return 0.0

    def get_summary_stats(self) -> Dict[str, float]:
        return {
            "num_nodes": self.g.number_of_nodes(),
            "num_edges": self.g.number_of_edges(),
            "density": float(nx.density(self.g)),
            "num_connected_components": float(nx.number_weakly_connected_components(self.g))
        }
