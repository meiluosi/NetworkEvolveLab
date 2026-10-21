import unittest
import numpy as np
from src.network_evolve.engine import CoEvolutionEngine
from src.network_evolve.semantic import VecWorld, CostLedger
from src.network_evolve.metrics import snapshot, cross_reach

class TestCoEvolutionExtensions(unittest.TestCase):
    def test_block_mechanism_prevents_reconnection(self):
        engine = CoEvolutionEngine(num_nodes=10, seed=42, block_diff=0.5, p_form=1.0)
        u, v = 0, 1
        engine.agents[u].belief = -0.9
        engine.agents[v].belief = 0.9
        engine.graph.g.add_edge(u, v, weight=0.5)
        engine.graph.g.add_edge(v, u, weight=0.5)
        
        # 强制 listener 接收 speaker 发言触发拉黑
        engine.step(compute_metrics=False)
        self.assertGreater(len(engine.blocked), 0)
        blocked_pair = list(engine.blocked)[0]
        listener, speaker = blocked_pair
        self.assertFalse(engine.graph.g.has_edge(listener, speaker))
        
        # 验证三元闭包不会对已拉黑节点重建连边
        engine.triadic_closure()
        self.assertFalse(engine.graph.g.has_edge(listener, speaker))

    def test_vec_world_semantic_backends(self):
        # 验证 DeGroot、HK 与 Proxy 均能稳定单步推进
        for be in ["degroot", "hk", "proxy"]:
            world = VecWorld(n=20, d=3, backend=be, eps=0.8, seed=42)
            init_edges = world.g.number_of_edges()
            metrics = world.run(rounds=2)
            self.assertIn("modularity", metrics)
            self.assertIn("polarization", metrics)
            self.assertIn("rupture_asymmetry", metrics)

    def test_cost_ledger_budget_exhaustion(self):
        ledger = CostLedger(budget_calls=2)
        ledger.charge("test1", 1)
        ledger.charge("test2", 1)
        with self.assertRaises(RuntimeError):
            ledger.charge("test3", 1)

    def test_cross_reach_metric(self):
        engine = CoEvolutionEngine(num_nodes=20, seed=42)
        reach = cross_reach(engine, n_sims=30, seed=42)
        self.assertGreaterEqual(reach, 0.0)
        self.assertLessEqual(reach, 1.0)

if __name__ == "__main__":
    unittest.main()
