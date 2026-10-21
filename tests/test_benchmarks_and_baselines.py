import unittest
import numpy as np
import networkx as nx
from src.network_evolve.datasets import (
    load_debate_benchmark, load_debate_network, load_snap_higgs, load_twitter_election, 
    load_reddit_convokit, load_birdwatch_community_notes, load_birdwatch_network
)
from src.network_evolve.baselines import (
    DeGrootModel, HegselmannKrauseModel, FriedkinJohnsenModel,
    GenerativeAgentsMemoryModel, GuEtAlPromptRewireModel, OursCoEvolutionModel
)

class TestBenchmarksAndBaselines(unittest.TestCase):
    def test_benchmark_loaders(self):
        # 1. DEBATE
        debates = load_debate_benchmark()
        self.assertGreaterEqual(len(debates), 5)
        self.assertIn("topic", debates[0])
        g_deb, ideo_deb = load_debate_network(num_nodes=30, seed=42)
        self.assertEqual(g_deb.number_of_nodes(), 30)
        self.assertEqual(len(ideo_deb), 30)
        self.assertGreater(g_deb.number_of_edges(), 10)
        
        # 2. SNAP Higgs
        g_higgs = load_snap_higgs(num_nodes=30, seed=42)
        self.assertEqual(g_higgs.number_of_nodes(), 30)
        self.assertGreater(g_higgs.number_of_edges(), 10)
        
        # 3. Twitter Election
        g_tw, ideo = load_twitter_election(num_nodes=30, seed=42)
        self.assertEqual(g_tw.number_of_nodes(), 30)
        self.assertEqual(len(ideo), 30)
        
        # 4. Reddit ConvoKit
        g_rd, topics = load_reddit_convokit(num_nodes=32, seed=42)
        self.assertEqual(g_rd.number_of_nodes(), 32)
        self.assertEqual(len(topics[0]), 3)
        
        # 5. Birdwatch
        bw = load_birdwatch_community_notes(num_nodes=30, seed=42)
        self.assertIn("notes", bw)
        self.assertGreater(len(bw["notes"]), 0)
        g_bw, ideo_bw = load_birdwatch_network(num_nodes=30, seed=42)
        self.assertEqual(g_bw.number_of_nodes(), 30)
        self.assertEqual(len(ideo_bw), 30)
        self.assertGreater(g_bw.number_of_edges(), 10)

    def test_baseline_models_step_and_metrics(self):
        g, ideo = load_twitter_election(num_nodes=30, seed=42)
        models = [
            DeGrootModel(g, ideo, seed=42),
            HegselmannKrauseModel(g, ideo, epsilon=0.5, seed=42),
            FriedkinJohnsenModel(g, ideo, stubbornness=0.3, seed=42),
            GenerativeAgentsMemoryModel(g, ideo, seed=42),
            GuEtAlPromptRewireModel(g, ideo, seed=42),
            OursCoEvolutionModel(g, ideo, seed=42)
        ]
        
        for m in models:
            res = m.run(rounds=3)
            self.assertIn("modularity_q", res)
            self.assertIn("polarization", res)
            self.assertIn("giant_frac", res)
            self.assertIn("num_edges", res)

if __name__ == "__main__":
    unittest.main()
