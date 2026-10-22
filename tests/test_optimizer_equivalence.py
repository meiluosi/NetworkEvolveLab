"""引擎性能优化的**逐位等价性**回归测试。

背景
----
为了让「网络规模敏感性」实验（N 到 1000）跑得动，我们对旗舰引擎做了三处
纯性能优化：缓存恒定的发言者权重向量、去掉标量 ``np.clip``、以及直接访问
networkx 的 ``adj`` 字典。任何这类改动都有**悄悄改变已有实验结果**的风险，
而本项目的全部结论都建立在那些结果之上。

因此这里用 ``tests/_baseline/`` 里保存的**优化前快照**做对照：
同一 seed、同一配置下，优化前后必须得到**完全相同**的边集、边权与全部指标。
快照一旦被更新，必须重跑 Phase 1b/3/4/5 与基准套件 —— 这条约束写在
``tests/_baseline/__init__.py`` 的文档字符串里。
"""
import unittest

from src.network_evolve.engine import CoEvolutionEngine as NewEngine
from tests._baseline.engine import CoEvolutionEngine as OldEngine

BASE = dict(base_tolerance=0.4, bridger_tolerance=0.7, p_form=0.1,
            theta_attract=0.4, heterogeneous=True, block_diff=1.2,
            rec_friction=1.0)


def _edge_snapshot(engine):
    g = engine.graph.g
    return sorted((int(u), int(v), round(float(d["weight"]), 12))
                  for u, v, d in g.edges(data=True))


def _belief_snapshot(engine):
    return [(i, round(float(a.belief), 12)) for i, a in sorted(engine.agents.items())]


class TestOptimizerEquivalence(unittest.TestCase):

    def _compare(self, seed, nodes, rounds, **kwargs):
        cfg = dict(BASE)
        cfg.update(kwargs)
        old = OldEngine(num_nodes=nodes, seed=seed, **cfg)
        new = NewEngine(num_nodes=nodes, seed=seed, **cfg)
        for _ in range(rounds):
            old.run_round()
            new.run_round()
        self.assertEqual(_edge_snapshot(old), _edge_snapshot(new),
                         f"seed={seed} N={nodes} rounds={rounds}: 边集/边权不一致")
        self.assertEqual(_belief_snapshot(old), _belief_snapshot(new),
                         f"seed={seed} N={nodes} rounds={rounds}: 信念轨迹不一致")

    def test_small_graph_several_seeds(self):
        for seed in (0, 3, 7):
            self._compare(seed, 40, 8)

    def test_medium_graph(self):
        self._compare(0, 120, 10)

    def test_heterogeneity_off(self):
        self._compare(1, 60, 8, heterogeneous=False)

    def test_different_block_threshold(self):
        self._compare(2, 60, 8, block_diff=1.0)

    def test_bridger_subsidy_and_stress(self):
        self._compare(4, 60, 8, bridger_subsidy=0.5, bridger_stress=True)

    def test_hub_limit(self):
        self._compare(5, 60, 8, hub_limit=0.3)


if __name__ == "__main__":
    unittest.main()
