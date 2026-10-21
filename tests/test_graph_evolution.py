import unittest
import numpy as np
from src.network_evolve.graph import DynamicSocialGraph
from src.network_evolve.agent import EvolvingAgent
from src.network_evolve.engine import CoEvolutionEngine

class TestNetworkEvolution(unittest.TestCase):
    def test_dynamic_graph_weight_and_unfollow(self):
        g = DynamicSocialGraph(num_nodes=5, initial_density=1.0, seed=42)
        initial_edges = g.g.number_of_edges()
        self.assertGreater(initial_edges, 0)
        
        # 强制将某条边的权重扣至负数，触发 unfollow 删边
        u, v = list(g.g.edges())[0]
        rem_w = g.update_edge_weight(u, v, delta=-2.0)
        self.assertEqual(rem_w, 0.0)
        self.assertFalse(g.g.has_edge(u, v))
        self.assertEqual(g.g.number_of_edges(), initial_edges - 1)

    def test_agent_cognitive_dissonance_decay(self):
        # 容忍度 0.4 的智能体
        agent = EvolvingAgent(agent_id=1, archetype="long_tail_responder", belief=-0.8, tolerance=0.4)
        # 受到极端对立者 (+0.8) 刺激
        res = agent.perceive_and_react(speaker_belief=0.8, speaker_archetype="hub_broadcaster")
        self.assertEqual(res["action"], "decay")
        self.assertLess(res["weight_delta"], 0.0)

    def test_engine_single_step(self):
        engine = CoEvolutionEngine(num_nodes=20, seed=42)
        step_res = engine.step()
        self.assertIn("speaker_id", step_res)
        self.assertIn("modularity_q", step_res)
        self.assertIn("polarization_var", step_res)

if __name__ == "__main__":
    unittest.main()
