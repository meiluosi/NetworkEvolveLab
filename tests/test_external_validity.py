"""Regression tests for the public real-stance benchmark assets."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network_evolve.datasets_stance import (  # noqa: E402
    load_polblogs_real_subgraph, polblogs_raw_path, polblogs_stats,
)
from src.network_evolve.datasets_polbooks import (  # noqa: E402
    load_polbooks_real_subgraph, raw_path as polbooks_raw_path,
)


class TestPolBlogsRealStance(unittest.TestCase):
    def test_polblogs_raw_is_present_and_labeled(self):
        self.assertIsNotNone(polblogs_raw_path())
        g, ideology = load_polblogs_real_subgraph(100, 0)
        self.assertEqual(g.number_of_nodes(), 100)
        self.assertGreater(g.number_of_edges(), 0)
        vals = np.array(list(ideology.values()), dtype=float)
        self.assertEqual(vals.size, 100)
        self.assertTrue(np.isfinite(vals).all())
        self.assertGreaterEqual(vals.min(), -1.0)
        self.assertLessEqual(vals.max(), 1.0)
        # The induced sample should retain both label signs.
        self.assertTrue((vals < 0).any())
        self.assertTrue((vals > 0).any())

    def test_polblogs_stats_are_finite(self):
        st = polblogs_stats(100, 0)
        self.assertEqual(st["num_nodes"], 100)
        self.assertGreaterEqual(st["left_frac"], 0.0)
        self.assertLessEqual(st["left_frac"], 1.0)
        self.assertTrue(np.isfinite(st["modularity_q"]))


class TestPolBooksRealStance(unittest.TestCase):
    def test_polbooks_raw_and_labels(self):
        self.assertIsNotNone(polbooks_raw_path())
        g, ideology = load_polbooks_real_subgraph(100, 0)
        self.assertEqual(g.number_of_nodes(), 100)
        self.assertGreater(g.number_of_edges(), 0)
        vals = np.array(list(ideology.values()), dtype=float)
        self.assertTrue(np.isfinite(vals).all())
        self.assertGreaterEqual(vals.min(), -1.0)
        self.assertLessEqual(vals.max(), 1.0)
        self.assertTrue((vals < 0).any())
        self.assertTrue((vals > 0).any())
