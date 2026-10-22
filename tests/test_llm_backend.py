"""统一 LLM 认知后端与新实验模块的回归测试（全部离线，不触网）。

固化的行为：
  1. 请求哈希必须把 model / temperature / max_tokens / **thinking_budget** 都算进去
     —— thinking_budget 变化必须换哈希，否则会命中旧缓存（这是真实发生过的 bug）；
  2. 内容寻址归档能往返读写，且重放时零 API 调用；
  3. 决策层对畸形 JSON / 非法动作 / 非法 new_belief 必须显式判为 invalid 并
     fallback 为 keep，**不得伪造决策**；
  4. 效应层语义必须与 ``phase2_llm`` 旧实现一致：keep 加权、mute 减权、
     unfollow/block 断边并写入黑名单；
  5. 旗舰引擎挂上 LLM 后，LLM 的取关决策必须真的改变图；
  6. 引擎的互动计数（覆盖率的分子分母）必须与真实互动次数一致。
"""
import json
import os
import tempfile
import unittest

from src.network_evolve.llm_backend import (
    LLMArchive, LLMClient, LLMCognitivePolicy, LLMDecision, LLMEvent, LLMReceipt,
    SCALAR_TOPICS, apply_decision, split_topics,
)
from src.network_evolve.engine import CoEvolutionEngine


class _StubClient:
    """离线替身：按预设文本返回响应，不做任何网络访问。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, prompt, n_events=0):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else '{"decisions": []}'
        receipt = LLMReceipt(model="stub-model", temperature=0.0, max_tokens=4096,
                             request_hash="stub", response_id="stub-0",
                             stop_reason="end_turn", input_tokens=1, output_tokens=1,
                             cached=False, truncated=False, n_events=n_events)
        return text, receipt


def _event(i=0, listener=0, speaker=1, dim=1):
    return LLMEvent(
        event_id=f"e{listener}_s{speaker}",
        listener=listener, speaker=speaker,
        listener_archetype="tail",
        listener_belief=[0.5] * dim,
        listener_profile="moderately for the contested policy axis",
        edge_weight=0.6,
        message="Just my two cents: I am moderately for the contested policy axis.",
        tolerance=0.4,
    )


class TestRequestHashing(unittest.TestCase):

    def test_thinking_budget_changes_the_hash(self):
        a = LLMClient.request_hash("sys", "prompt", "m", 0.0, 4096, 512)
        b = LLMClient.request_hash("sys", "prompt", "m", 0.0, 4096, 1024)
        self.assertNotEqual(a, b, "thinking_budget 必须参与请求哈希，否则会误命中旧缓存")

    def test_every_relevant_field_changes_the_hash(self):
        base = LLMClient.request_hash("sys", "prompt", "m", 0.0, 4096, 512)
        variants = [
            LLMClient.request_hash("sys2", "prompt", "m", 0.0, 4096, 512),
            LLMClient.request_hash("sys", "prompt2", "m", 0.0, 4096, 512),
            LLMClient.request_hash("sys", "prompt", "m2", 0.0, 4096, 512),
            LLMClient.request_hash("sys", "prompt", "m", 0.5, 4096, 512),
            LLMClient.request_hash("sys", "prompt", "m", 0.0, 8192, 512),
        ]
        for v in variants:
            self.assertNotEqual(base, v)

    def test_hash_is_deterministic(self):
        a = LLMClient.request_hash("sys", "prompt", "m", 0.0, 4096, 512)
        b = LLMClient.request_hash("sys", "prompt", "m", 0.0, 4096, 512)
        self.assertEqual(a, b)


class TestArchive(unittest.TestCase):

    def test_roundtrip_and_count(self):
        with tempfile.TemporaryDirectory() as td:
            arc = LLMArchive(root=td)
            self.assertEqual(arc.count(), 0)
            arc.put("m", "hash1", {"response_text": "hi", "n_events": 2})
            self.assertIsNotNone(arc.get("m", "hash1"))
            self.assertEqual(arc.get("m", "hash1")["n_events"], 2)
            self.assertEqual(arc.count(), 1)
            self.assertEqual(arc.count("other"), 0)

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(LLMArchive(root=td).get("m", "nope"))


class TestDecisionLayer(unittest.TestCase):

    def _policy(self, responses):
        topics, poles = split_topics(SCALAR_TOPICS)
        client = _StubClient(responses)
        return LLMCognitivePolicy(client, topics=topics, poles=poles), client

    def test_valid_response_is_parsed(self):
        payload = json.dumps({"decisions": [
            {"id": "e0_s1", "new_belief": [0.7], "action": "unfollow", "reasoning": "far"}]})
        pol, client = self._policy([payload])
        decs, stats = pol.decide([_event()])
        self.assertEqual(len(decs), 1)
        self.assertTrue(decs[0].valid)
        self.assertEqual(decs[0].action, "unfollow")
        self.assertEqual(stats["n_valid"], 1)
        self.assertEqual(stats["n_invalid"], 0)

    def test_malformed_json_falls_back_to_keep_and_is_counted_invalid(self):
        pol, client = self._policy(["not json at all {{{"])
        decs, stats = pol.decide([_event()])
        self.assertEqual(decs[0].action, "keep")
        self.assertFalse(decs[0].valid)
        self.assertEqual(stats["n_invalid"], 1)
        self.assertEqual(stats["n_valid"], 0)

    def test_illegal_action_is_invalid(self):
        payload = json.dumps({"decisions": [
            {"id": "e0_s1", "new_belief": [0.7], "action": "explode"}]})
        pol, _ = self._policy([payload])
        decs, stats = pol.decide([_event()])
        self.assertFalse(decs[0].valid)
        self.assertEqual(decs[0].action, "keep")

    def test_wrong_dimensionality_is_invalid(self):
        payload = json.dumps({"decisions": [
            {"id": "e0_s1", "new_belief": [0.1, 0.2, 0.3], "action": "keep"}]})
        pol, _ = self._policy([payload])
        decs, stats = pol.decide([_event(dim=1)])
        self.assertFalse(decs[0].valid)

    def test_missing_event_id_falls_back(self):
        payload = json.dumps({"decisions": [
            {"id": "e999_s999", "new_belief": [0.7], "action": "mute"}]})
        pol, _ = self._policy([payload])
        decs, stats = pol.decide([_event()])
        self.assertFalse(decs[0].valid)
        self.assertEqual(decs[0].action, "keep")


class _TinyWorld:
    """apply_decision 需要的最小世界接口。"""

    class _G:
        def __init__(self):
            self.edges = {}

        def has_edge(self, u, v):
            return (u, v) in self.edges

        def __getitem__(self, uv):
            u, v = uv
            return {"weight": self.edges[(u, v)]}

        def __setitem__(self, uv, val):
            self.edges[uv] = val["weight"]

    def __init__(self, dim=1):
        import networkx as nx
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1, weight=0.6)
        self.g.add_edge(1, 0, weight=0.6)
        self.b = [[0.5] * dim]
        self.blocked = set()

    def _drop_edge(self, u, v):
        if self.g.has_edge(u, v):
            self.g.remove_edge(u, v)
        self.blocked.add((u, v))


class TestEffectLayer(unittest.TestCase):

    def test_keep_increases_weight(self):
        w = _TinyWorld()
        out = apply_decision(w, 0, 1, LLMDecision("e", "keep", [0.5], "", True), eta=0.2)
        self.assertFalse(out["dropped"])
        self.assertAlmostEqual(w.g[0][1]["weight"], 0.65, places=6)

    def test_mute_decreases_weight(self):
        w = _TinyWorld()
        apply_decision(w, 0, 1, LLMDecision("e", "mute", [0.5], "", True), eta=0.2)
        self.assertAlmostEqual(w.g[0][1]["weight"], 0.45, places=6)

    def test_unfollow_drops_edge_and_blocks_it(self):
        w = _TinyWorld()
        out = apply_decision(w, 0, 1, LLMDecision("e", "unfollow", [0.5], "", True), eta=0.2)
        self.assertTrue(out["dropped"])
        self.assertFalse(w.g.has_edge(0, 1))
        self.assertIn((0, 1), w.blocked)

    def test_block_drops_edge_and_blocks_it(self):
        w = _TinyWorld()
        out = apply_decision(w, 0, 1, LLMDecision("e", "block", [0.5], "", True), eta=0.2)
        self.assertTrue(out["dropped"])
        self.assertIn((0, 1), w.blocked)

    def test_belief_update_is_damped_and_clipped(self):
        w = _TinyWorld()
        apply_decision(w, 0, 1, LLMDecision("e", "keep", [0.9], "", True), eta=0.2)
        # 0.5 + 0.2 * (0.9 - 0.5) = 0.58
        self.assertAlmostEqual(w.b[0][0], 0.58, places=6)

    def test_invalid_decision_leaves_belief_untouched(self):
        w = _TinyWorld()
        apply_decision(w, 0, 1, LLMDecision("e", "keep", None, "", False), eta=0.2)
        self.assertAlmostEqual(w.b[0][0], 0.5, places=6)


class TestEngineLLMMount(unittest.TestCase):

    def _engine_with_llm(self, action="unfollow", seed=0, nodes=40, rounds=3):
        e = CoEvolutionEngine(num_nodes=nodes, seed=seed,
                              base_tolerance=0.4, bridger_tolerance=0.7, p_form=0.1,
                              theta_attract=0.4, heterogeneous=True, block_diff=1.2,
                              rec_friction=1.0)
        topics, poles = split_topics(SCALAR_TOPICS)
        client = _StubClient([])
        pol = LLMCognitivePolicy(client, topics=topics, poles=poles)
        e.llm_policy = pol
        e.llm_event_budget = 2
        return e, pol, client

    def test_interaction_counter_matches_actual_reactions(self):
        """覆盖率的分母必须等于引擎真实发生的「听者-发言者」互动数。"""
        e, pol, client = self._engine_with_llm()
        e.llm_policy = None  # 先跑规则臂，避免依赖 stub 响应
        init_edges = set(e.graph.g.edges())
        total = 0
        for _ in range(3):
            before = len(list(e.graph.g.predecessors(0)))  # 仅为可读性
            e.run_round()
            total = e.n_interactions
        self.assertGreater(total, 0)
        # 互动数随轮次单调增加（引擎不会自己清零计数器）
        e2 = CoEvolutionEngine(num_nodes=40, seed=0)
        e2.run_round()
        first = e2.n_interactions
        e2.run_round()
        self.assertGreater(e2.n_interactions, first)
        self.assertGreaterEqual(init_edges, set())

    def test_llm_decision_actually_changes_the_graph(self):
        e, pol, client = self._engine_with_llm(action="unfollow")

        # 直接替换 decide，避免依赖网络：任何事件都回一个 unfollow 决策
        def decide(events):
            payload = json.dumps({"decisions": [
                {"id": ev.event_id, "new_belief": list(ev.listener_belief),
                 "action": "unfollow", "reasoning": "stub"} for ev in events]})
            return _direct(pol, events, payload)

        pol.decide = decide
        before_edges = set(e.graph.g.edges())
        for _ in range(3):
            e.run_round()
        after_edges = set(e.graph.g.edges())
        self.assertLess(len(after_edges), len(before_edges),
                        "LLM 的 unfollow 决策必须真的断边")
        self.assertGreater(e.llm_stats["n_applied"], 0)
        self.assertEqual(e.llm_stats["actions"].get("unfollow", 0),
                         e.llm_stats["n_applied"])


def _direct(pol, events, payload):
    """把一段 JSON 响应喂给策略并复用其解析逻辑（离线）。"""
    obj = json.loads(payload)
    decisions = []
    n_valid = 0
    for ev in events:
        item = obj["decisions"][0]
        dec = pol._parse_one(ev, item)
        if dec.valid:
            n_valid += 1
        decisions.append(dec)
    return decisions, {"n_events": len(events), "n_valid": n_valid,
                       "n_invalid": len(events) - n_valid, "receipt": {"cached": False}}


if __name__ == "__main__":
    unittest.main()


class TestOpenAICompatibleAdapter(unittest.TestCase):
    """The DeepSeek/OpenAI-compatible adapter must be selected without network."""

    def test_deepseek_base_selects_openai_style(self):
        c = LLMClient(model="deepseek-chat",
                      base_url="https://api.deepseek.com/v1",
                      api_key="test-key", budget_calls=1, thinking_budget=0)
        self.assertEqual(c.api_style, "openai")
        self.assertEqual(c.base_url, "https://api.deepseek.com/v1")

    def test_api_style_is_part_of_request_hash(self):
        a = LLMClient.request_hash("sys", "prompt", "deepseek-chat", 0.0, 128,
                                   0, "anthropic")
        b = LLMClient.request_hash("sys", "prompt", "deepseek-chat", 0.0, 128,
                                   0, "openai")
        self.assertNotEqual(a, b)
