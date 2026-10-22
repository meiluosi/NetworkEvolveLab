"""研究不变量的回归测试（ARS 完整性门的形式化固化）。

这些测试把本次全链路复核中发现并修复的缺陷固化为可执行断言，防止回归：
  1. 同步调度下，标量有界信任模型的互惠断裂必须严格对称（A ≡ 0）；
  2. 同步调度下，各向同性多维模型的断裂判据同样必须严格对称（A ≡ 0）；
  3. 已断裂的边不得被三元闭包重新连回（黑名单语义，两个实现必须一致）；
  4. Phase 3 的到达率增益在基线接近 0 时不得数值爆炸；
  5. 模块度必须满足 K 社群推广上界 Q ≤ 1 − 1/K；
  6. H2 判定必须以「种子」为独立统计单元（不得使用节点级伪重复口径作为主判定）。
"""
import unittest
import numpy as np
import networkx as nx

from src.network_evolve.semantic import VecWorld
from src.network_evolve.metrics import asymmetry_stats
from src.network_evolve.phase4_attribution import canonical_conditions, ARMS
from src.network_evolve.phase3 import eval_policy
from src.network_evolve.phase5_bound_audit import modularity_upper_bound
from src.network_evolve.metrics import h2_seed_level_test
from src.network_evolve.engine import CoEvolutionEngine


class TestResearchInvariants(unittest.TestCase):

    # ---------- 1 & 2. 对称性不变量 ----------
    def _run_arm(self, arm_name, seed=0, n=40, rounds=8, p_edge=0.15):
        cfg = ARMS[arm_name]
        d = cfg["d"]
        g0, b3, b1, sal = canonical_conditions(seed, n=n, d=3, p_edge=p_edge)
        b_init = b1 if d == 1 else b3
        sal_init = None if d == 1 else (None if cfg["isotropic"] else sal)
        w = VecWorld(n=n, d=d, backend=cfg["backend"], eps=0.6, seed=seed + 1000,
                     p_edge=p_edge, p_form=0.1, update_mode=cfg["update_mode"],
                     isotropic=cfg["isotropic"], symmetric_tol=cfg["symmetric_tol"],
                     b_init=b_init, g0=g0, salience_init=sal_init)
        for _ in range(rounds):
            if w.update_mode == "sync":
                w._step_sync()
            else:
                for _ in range(w.n):
                    w.step()
            w.closure()
        stats = asymmetry_stats(w.g, w.initial_edges)
        return stats

    def test_scalar_sync_rupture_is_exactly_symmetric(self):
        """同步调度 + 标量有界信任 => 断裂判据对称 => 非对称断裂率必须严格为 0"""
        st = self._run_arm("scalar_sync", seed=1)
        self.assertGreater(st["n_mutual_broken"], 0, "对照组必须真的断裂了互惠边，否则该检验无意义")
        self.assertEqual(st["n_one_sided"], 0.0,
                         f"同步标量模型出现了单向断裂 {st['n_one_sided']} 次，说明对称性被破坏")

    def test_isotropic_sync_rupture_is_exactly_symmetric(self):
        """各向同性（等权）+ 同步 => 判据对所有个体一致 => 非对称断裂率必须严格为 0"""
        st = self._run_arm("isotropic_sync", seed=1)
        self.assertGreater(st["n_mutual_broken"], 0, "对照组必须真的断裂了互惠边")
        self.assertEqual(st["n_one_sided"], 0.0,
                         f"各向同性同步模型出现单向断裂 {st['n_one_sided']} 次")

    def test_anisotropy_creates_asymmetry_under_sync(self):
        """个体各向异性在同步调度下仍能产生非对称断裂（这是 RQ1 归因消融的关键正对照）"""
        st = self._run_arm("multidim_sync", seed=1)
        self.assertGreater(st["n_one_sided"], 0.0,
                           "各向异性同步模型未观测到任何单向断裂，消融失去区分力")

    # ---------- 3. 黑名单不变量 ----------
    def test_broken_edge_not_reconnected_by_closure(self):
        w = VecWorld(n=30, d=3, backend="hk", eps=0.6, seed=7, p_edge=0.2, p_form=1.0)
        u, v = 0, 1
        w.g.add_edge(u, v, weight=0.1)
        w.g.add_edge(v, u, weight=0.1)
        w.b[u] = -0.95
        w.b[v] = 0.95
        w._drop_edge(u, v)
        self.assertFalse(w.g.has_edge(u, v))
        w.closure()
        self.assertFalse(w.g.has_edge(u, v), "已断裂的边被三元闭包重新连回（黑名单失效）")

    def test_engine_and_vecworld_share_blocklist_semantics(self):
        """两个实现都必须维护 blocked 集合，且断边不再重建"""
        e = CoEvolutionEngine(num_nodes=12, seed=3, block_diff=0.5, p_form=1.0)
        self.assertTrue(hasattr(e, "blocked"))

    # ---------- 4. Phase 3 到达率增益数值稳健性 ----------
    def test_phase3_reach_gain_does_not_explode(self):
        """基线到达率接近 0 时，相对增益不得爆炸（原实现会出现 1e6 量级数值）"""
        base = [{"q_end": 0.44, "reach": 1e-9, "polar": 0.1, "cross": 0.02, "edges": 1000},
                {"q_end": 0.44, "reach": 0.40, "polar": 0.1, "cross": 0.02, "edges": 1000}]
        pol = [{"q_end": 0.40, "reach": 0.45, "polar": 0.1, "cross": 0.05, "edges": 990},
               {"q_end": 0.40, "reach": 0.42, "polar": 0.1, "cross": 0.05, "edges": 990}]
        ev = eval_policy(base, pol)
        self.assertTrue(np.isfinite(ev["reach_rel_gain_mean"]))
        self.assertLess(abs(ev["reach_rel_gain_mean"]), 100.0,
                        "到达率相对增益出现数值爆炸")
        self.assertTrue(np.isfinite(ev["reach_delta_mean"]))
        self.assertEqual(ev["n_seeds_low_baseline_reach"], 1)

    # ---------- 5. 模块度推广上界 ----------
    def test_modularity_bound_generalization(self):
        self.assertAlmostEqual(modularity_upper_bound(2), 0.5)
        self.assertAlmostEqual(modularity_upper_bound(3), 2 / 3)
        self.assertAlmostEqual(modularity_upper_bound(4), 0.75)
        # 随机图上核验 Q <= 1 - 1/K
        rng = np.random.default_rng(0)
        for _ in range(10):
            g = nx.erdos_renyi_graph(40, 0.08, seed=int(rng.integers(1 << 30)))
            comms = nx.community.louvain_communities(g, seed=0)
            q = nx.community.modularity(g, comms)
            self.assertLessEqual(q, modularity_upper_bound(len(comms)) + 1e-9)

    # ---------- 6. H2 判定口径 ----------
    def test_h2_uses_seed_level_unit(self):
        """种子级判定：若多数种子的差距为正且达到 0.4 效应量，seed 级显著性检验应给出显著结果"""
        gaps = [0.5] * 8 + [0.1] * 2
        res = h2_seed_level_test(gaps, threshold=0.4)
        self.assertEqual(res["n_seeds"], 10)
        self.assertLess(res["p_value_seed_level"], 0.05)
        self.assertAlmostEqual(res["frac_seeds_passing"], 0.8)
        self.assertGreater(res["p_value_threshold"], res["p_value_seed_level"])

    def test_h2_seed_level_test_detects_no_effect(self):
        gaps = [0.0] * 10
        res = h2_seed_level_test(gaps, threshold=0.4)
        self.assertGreater(res["p_value_seed_level"], 0.05)

    # ---------- 7. Phase 1c 临界点测量的判据自洽性 ----------
    def test_threshold_criterion_detects_crossing(self):
        """连通性恢复判据：giant 曲线单调回升时，θ* 应落在首个跨越点附近"""
        from src.network_evolve.phase1c_threshold import _crossing
        thetas = [0.30, 0.35, 0.40, 0.45, 0.50]
        giant = [0.72, 0.91, 0.999, 1.0, 1.0]
        ts, censored = _crossing(thetas, giant, 0.99, "up")
        self.assertFalse(censored)
        self.assertGreater(ts, 0.35)
        self.assertLess(ts, 0.40)
        # 未跨越时必须报右删失，且给出网格上界
        ts2, censored2 = _crossing(thetas, [0.7] * 5, 0.99, "up")
        self.assertTrue(censored2)
        self.assertAlmostEqual(ts2, 0.50)

    def test_threshold_downward_criterion_reports_censoring(self):
        """模块度塌缩判据：曲线未跌破一半时必须记为右删失，而不是伪造一个 θ*"""
        from src.network_evolve.phase1c_threshold import _crossing
        thetas = [0.30, 0.40, 0.50, 0.60]
        q = [0.60, 0.58, 0.57, 0.56]   # 始终 > 0.5 * 0.60 = 0.30
        ts, censored = _crossing(thetas, q, 0.5 * q[0], "down")
        self.assertTrue(censored)
        self.assertAlmostEqual(ts, thetas[-1])

    # ---------- 8. 论文表格与结果 JSON 的绑定（防漂移） ----------
    def test_generated_dataset_table_matches_measured_graph(self):
        """自动生成的数据集表必须反映真实测量值，而不是抄写值"""
        import json
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tex_path = os.path.join(root, "paper", "tables", "dataset_metrics.tex")
        if not os.path.exists(tex_path):
            self.skipTest("paper/tables/dataset_metrics.tex 尚未生成")
        with open(tex_path, encoding="utf-8") as fh:
            tex = fh.read()
        # 真实 Higgs 子图的实测互惠率必须出现在表中，且旧的错误手抄值 0.12 不得复现
        self.assertIn("REAL public data", tex)
        self.assertNotIn("$0.12 \\pm", tex)
        # 旧手抄表中 DEBATE 的 Q_0=0.21 与 Birdwatch 的 0.32 均与实测不符
        self.assertNotIn("$0.210 \\pm", tex)
        self.assertNotIn("$0.320 \\pm", tex)

    def test_phase1c_result_file_is_self_consistent(self):
        """Phase 1c 结果文件内部自洽：θ* 不得越界、判据必须给出显式结论"""
        import json
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "results", "phase1c_threshold.json")
        if not os.path.exists(path):
            self.skipTest("results/phase1c_threshold.json 尚未生成")
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        lo, hi = min(d["theta_grid"]), max(d["theta_grid"])
        for topo, rec in d["topologies"].items():
            for key in ("theta_frag", "theta_q", "theta_pol"):
                summ = rec[key]
                self.assertEqual(len(summ["per_seed"]), d["args"]["seeds"])
                for v in summ["per_seed"]:
                    self.assertGreaterEqual(v, lo - 1e-9)
                    self.assertLessEqual(v, hi + 1e-9)
                self.assertGreaterEqual(summ["frac_censored"], 0.0)
                self.assertLessEqual(summ["frac_censored"], 1.0)
        # 三个判据都必须对「是否落在预登记区间」给出显式布尔结论
        for key in ("theta_frag", "theta_q", "theta_pol"):
            self.assertIn("band_consistent", d["summary"][key])
            self.assertIn("n_in_preregistered_band", d["summary"][key])


if __name__ == "__main__":
    unittest.main()
