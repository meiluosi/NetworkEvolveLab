"""Phase 6：旗舰引擎挂载**真实 LLM 认知层**的受控对照实验。

为什么需要这个实验
------------------
复核发现的发表阻断项：旗舰 ``CoEvolutionEngine`` 原本只有**标量规则启发式**认知
（``EvolvingAgent.perceive_and_react``），「LLM 驱动拓扑」这一标题级主张实际上
只在一个 3 种子的侧实验里成立。本实验把真实 LLM 接进旗舰引擎，
用**同一初始条件**（同 seed ⇒ 同初始图与同初始信念）做规则 vs LLM 的配对对照。

设计要点
--------
1. **闭环**：LLM 的决策（keep / mute / unfollow / block）在轮内真实落地 ——
   取关会真的断边并进入黑名单，静音会真的衰减边权。
2. **每轮一次批量调用**：引擎的一轮含 N 个单步，若每步问一次 LLM，调用数会被
   放大 N 倍（实测 15 s/次），既昂贵又让「LLM 覆盖比例」失去可解释性。
   因此在轮首按发言分布采样 K 个二元组，一次批量调用取得决策，轮内命中即采用。
   **覆盖比例会被显式报告**：LLM 实际落地决策的互动数 / 引擎发生的全部互动数
   （不是"被问到的事件数"——有些计划事件因听众在轮内被拉黑或边先被切断而不会发生互动），
   不隐瞒混合后端的性质。
3. **配对比较**：规则臂与 LLM 臂共享初始条件，按种子配对求差，
   统计量以**种子**为独立单元（避免节点级伪重复）。
4. **完整收据**：模型标识、temperature、input/output/thinking token 数、
   解析失败率、动作分布全部落盘，补上此前真实 LLM 臂缺失的收据字段。

用法：
    python -m src.network_evolve.phase6_llm_engine --seeds 5 --rounds 30 \
        --nodes 60 --llm-events 6 --out results/phase6_llm_engine.json
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from .engine import CoEvolutionEngine
from .metrics import engine_metrics, init_incident_edges, retention_by_node
from .expkit import mean_ci
from .llm_backend import (LLMClient, LLMCognitivePolicy, SCALAR_TOPICS,
                           split_topics)

BASE_CFG = dict(base_tolerance=0.4, bridger_tolerance=0.7, p_form=0.1,
                 theta_attract=0.4, heterogeneous=True, block_diff=1.2,
                 rec_friction=1.0)


def _checkpoint(out, path):
    """把当前进度写到磁盘（原子替换），避免长跑被网络故障打断后丢失结果。"""
    tmp = path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_arm(arm: str, seed: int, nodes: int, rounds: int,
            llm_events: int, client: LLMClient, policy: LLMCognitivePolicy):
    """跑一个实验臂，返回 (指标, LLM 统计)。"""
    e = CoEvolutionEngine(num_nodes=nodes, seed=seed, **BASE_CFG)
    if arm == "llm_hybrid":
        e.llm_policy = policy
        e.llm_event_budget = llm_events
    init_edges = set(e.graph.g.edges())
    incident = init_incident_edges(e)
    for _ in range(rounds):
        e.run_round()
    m = engine_metrics(e, init_edges)

    # LLM 决策的动作分布（真实落地到拓扑的那些）
    stats_llm = dict(e.llm_stats)
    stats_llm["actions"] = dict(stats_llm.get("actions", {}))
    stats_llm["n_interactions"] = int(e.n_interactions)
    ret = retention_by_node(e, incident)
    bridger = [v for i, v in ret.items()
               if e.agents[i].archetype == "active_bridger"]
    other = [v for i, v in ret.items()
             if e.agents[i].archetype != "active_bridger"]
    m["bridger_retention_mean"] = float(np.mean(bridger)) if bridger else float("nan")
    m["other_retention_mean"] = float(np.mean(other)) if other else float("nan")
    m["n_bridger"] = len(bridger)
    m["n_interactions"] = int(e.n_interactions)
    return m, stats_llm


def main():
    ap = argparse.ArgumentParser(description="Phase 6: real-LLM cognitive layer vs rule backend")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--llm-events", type=int, default=6,
                    help="每轮交给 LLM 决策的认知事件数（每轮一次批量调用）")
    ap.add_argument("--model", type=str, default="MiniMax-M3.1-Flash")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking-budget", type=int, default=512)
    ap.add_argument("--budget-calls", type=int, default=1000)
    ap.add_argument("--out", type=str, default="results/phase6_llm_engine.json")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    topics, poles = split_topics(SCALAR_TOPICS)
    client = LLMClient(model=args.model, temperature=0.0, max_tokens=args.max_tokens,
                       budget_calls=args.budget_calls, thinking_budget=args.thinking_budget)
    policy = LLMCognitivePolicy(client, topics=topics, poles=poles)

    print(f"=== Phase 6: 真 LLM 认知层 vs 规则后端 "
          f"(Seeds={args.seeds}, Rounds={args.rounds}, N={args.nodes}, "
          f"LLM events/round={args.llm_events}) ===")
    print(f"model={args.model}  max_tokens={args.max_tokens}  "
          f"thinking_budget={args.thinking_budget}")

    out = {
        "args": vars(args),
        "design": {
            "arms": ["rule", "llm_hybrid"],
            "pairing": "同 seed ⇒ 同初始图与同初始信念（配对比较）",
            "statistical_unit": "seed",
            "llm_mode": "每轮一次批量调用，轮内命中的二元组采用 LLM 决策（闭环）",
            "caveat": "llm_hybrid 是混合后端：未被 LLM 覆盖的互动仍走规则后端；"
                      "覆盖比例在 llm_coverage 中显式报告",
        },
        "arms": {},
        "contrasts": {},
        "verdict": {},
    }

    per_seed = {"rule": [], "llm_hybrid": []}
    llm_stats_acc = {"n_events": 0, "n_valid": 0, "n_invalid": 0,
                      "n_calls": 0, "n_cache_hits": 0, "n_rounds_with_llm": 0,
                      "n_applied": 0, "n_interactions": 0, "actions": {}}
    for s in seeds:
        row = {}
        for arm in ("rule", "llm_hybrid"):
            m, st = run_arm(arm, s, args.nodes, args.rounds, args.llm_events,
                            client, policy)
            per_seed[arm].append(m)
            row[arm] = m
            if arm == "llm_hybrid":
                for k in ("n_events", "n_valid", "n_invalid", "n_calls",
                          "n_cache_hits", "n_rounds_with_llm", "n_applied",
                          "n_interactions"):
                    llm_stats_acc[k] += st.get(k, 0)
                for a, c in (st.get("actions") or {}).items():
                    llm_stats_acc["actions"][a] = llm_stats_acc["actions"].get(a, 0) + c
                row["llm_stats"] = st
        out["arms"] = out["arms"] or {}
        out.setdefault("per_seed", {})[str(s)] = row
        # 增量落盘：LLM 调用偶发网络故障时不至于丢掉已完成的种子
        _checkpoint(out, args.out)
        print(f"  seed {s}: rule Q={row['rule']['modularity_q']:.3f} "
              f"| llm Q={row['llm_hybrid']['modularity_q']:.3f} "
              f"| llm 事件 {row['llm_stats']['n_events']} "
              f"(有效 {row['llm_stats']['n_valid']}, 落地 {row['llm_stats']['n_applied']})")

    # —— 汇总 ——
    def agg(arm, key):
        arr = np.array([m[key] for m in per_seed[arm]], dtype=float)
        mean, half = mean_ci(arr)
        return {"mean": float(mean), "ci95": float(half),
                "per_seed": [float(x) for x in arr]}

    summary = {}
    for arm in ("rule", "llm_hybrid"):
        summary[arm] = {k: agg(arm, k) for k in
                        ("modularity_q", "polarization", "giant_frac",
                         "small_comm_frac", "rupture_asymmetry", "num_comms",
                         "num_edges", "bridger_retention_mean",
                         "other_retention_mean", "n_interactions")}

    contrasts = {}
    for key in ("modularity_q", "polarization", "giant_frac", "rupture_asymmetry",
                "num_edges", "bridger_retention_mean"):
        a = np.array([m[key] for m in per_seed["llm_hybrid"]], dtype=float)
        b = np.array([m[key] for m in per_seed["rule"]], dtype=float)
        d = a - b
        mean, half = mean_ci(d)
        if np.allclose(d, d[0]):
            # 退化情形：所有种子的配对差完全相同（例如 LLM 臂从未改变该指标，
            # 差值全为 0）。此时配对 t 检验无定义（0/0），p 会变成 NaN 并污染
            # 论文表格，因此显式处理：差值恒为 0 ⇒ 无差异证据（p = 1）。
            t_stat, p_val = 0.0, (1.0 if d[0] == 0 else 0.0)
        else:
            t_stat, p_val = stats.ttest_rel(a, b)
            t_stat, p_val = float(t_stat), float(p_val)
        contrasts[key] = {
            "mean_diff_llm_minus_rule": float(mean), "ci95": [float(half), -float(half)],
            "p_value_paired_t": float(p_val),
            "t_statistic": float(t_stat),
            "degenerate_zero_variance": bool(np.allclose(d, d[0])),
            "n_positive": int(np.sum(d > 0)), "n_pairs": int(d.size),
            "per_seed_diff": [float(x) for x in d],
        }

    total_events = llm_stats_acc["n_events"]
    total_applied = llm_stats_acc["n_applied"]
    total_interactions = llm_stats_acc["n_interactions"]
    # 真实覆盖率：LLM 实际落地决策的互动数 / 引擎发生的全部互动数
    coverage = total_applied / total_interactions if total_interactions else 0.0
    # 计划命中率：被问到的事件中真正在轮内发生互动的比例
    plan_hit = total_applied / total_events if total_events else 0.0
    rule_interactions = sum(m.get("n_interactions", 0) for m in per_seed["rule"])
    out["arms"] = summary
    out["contrasts"] = contrasts
    ledger = client.ledger()
    # 本实验**去重后的请求数**才是它真实消耗的 API 调用次数：
    # 每个不同的请求哈希恰好对应一次 API 调用，而重跑时全部命中归档。
    # 只报「本次运行的 api_calls」会在重放时变成 0，掩盖真实花费。
    ledger["distinct_requests_in_experiment"] = len(client.seen_hashes)
    ledger["total_api_calls_for_experiment"] = len(client.seen_hashes)
    ledger["archive_entries_for_model"] = int(client.archive.count(args.model))
    ledger["cost_note"] = (
        "distinct_requests_in_experiment 是本实验内容寻址请求的去重条数，"
        "每条恰好对应一次真实 API 调用；重跑时全部命中归档（零成本、逐位可复现）。")
    out["llm_ledger"] = ledger
    out["llm_stats_total"] = llm_stats_acc
    out["llm_coverage"] = {
        "llm_asked_events": total_events,
        "llm_decided_events": total_applied,
        "valid_events": llm_stats_acc["n_valid"],
        "invalid_events": llm_stats_acc["n_invalid"],
        "engine_interactions_llm_arm": total_interactions,
        "engine_interactions_rule_arm": int(rule_interactions),
        "coverage_of_interactions": coverage,
        "plan_hit_rate": plan_hit,
        "parse_success_rate": (llm_stats_acc["n_valid"] / total_events
                               if total_events else 0.0),
        "rounds_with_llm": llm_stats_acc["n_rounds_with_llm"],
        "action_distribution": llm_stats_acc["actions"],
        "note": "coverage_of_interactions = LLM 实际落地决策的互动数 / LLM 臂引擎全部互动数；"
                "llm_hybrid 是混合后端，未被 LLM 覆盖的互动仍走规则后端（已显式披露）",
    }
    out["verdict"] = {
        "llm_changes_modularity": bool(abs(contrasts["modularity_q"]["mean_diff_llm_minus_rule"]) > 0.01),
        "llm_changes_rupture_asymmetry": bool(
            abs(contrasts["rupture_asymmetry"]["mean_diff_llm_minus_rule"]) > 0.01),
        "modularity_diff_significant": bool(contrasts["modularity_q"]["p_value_paired_t"] < 0.05),
        "rupture_diff_significant": bool(contrasts["rupture_asymmetry"]["p_value_paired_t"] < 0.05),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n--- 配对对比 (LLM − 规则，种子级) ---")
    for k, v in contrasts.items():
        print(f"  {k:<24} Δ={v['mean_diff_llm_minus_rule']:+.4f} "
              f"p={v['p_value_paired_t']:.3g}  ({v['n_positive']}/{v['n_pairs']} 正向)")
    print(f"\nLLM 收据: {client.ledger()['api_calls']} 次 API 调用, "
          f"cache_hits={client.ledger()['cache_hits']}, tokens={client.ledger()['totals']}")
    print(f"解析成功率: {out['llm_coverage']['parse_success_rate']:.1%} "
          f"({llm_stats_acc['n_valid']}/{total_events})")
    print(f"互动覆盖率: {coverage:.2%} "
          f"(LLM 落地 {total_applied} / 引擎互动 {total_interactions}); "
          f"计划命中率 {plan_hit:.1%}")
    print(f"LLM 动作分布: {llm_stats_acc['actions']}")
    print(f"[Done] {args.out}")


if __name__ == "__main__":
    main()
