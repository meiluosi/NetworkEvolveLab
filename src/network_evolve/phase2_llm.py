"""Phase 2 真实 LLM 认知臂（小规模先导实验，批量请求 + 缓存 + 成本记账）。

流程（每轮 1 次批量 LLM 调用 = 1 次“API 调用”）:
  plan   : 抽取本轮发言事件，生成 req_r{r}.json（含听众人设 + 发言文本）
  [LLM]  : 外部 LLM 读取 req 文件，写出 resp_r{r}.json
  apply  : 校验并应用响应（信念位移经阻尼 eta，动作映射到边操作），预算计数 +1
  finish : 输出最终拓扑/认知指标
  compare: 对同种子的 DeGroot / HK / proxy 臂用相同事件结构运行，与 LLM 臂并列
"""
import argparse, json, os, pickle, hashlib
import numpy as np
from .semantic import VecWorld, CostLedger
# —— 统一 LLM 认知后端（本次复核重构）——
# 决策逻辑（prompt 构造 / JSON 解析 / 决策落地）统一到 llm_backend，
# 本模块只保留「离线归档工作流」（plan / apply / finish）与事件生成。
from .llm_backend import (LLMDecision, apply_decision, archetype_of,
                          message_text, split_topics, verbalize,
                          DEFAULT_TOPICS)

TOPICS = DEFAULT_TOPICS
#: 议题图例的拆分形式（topic 列表 + (负极, 正极) 列表），与统一后端一致
TOPIC_NAMES, TOPIC_POLES = split_topics(TOPICS)
ETA = 0.2          # 阻尼学习率（ARCHITECTURE.md: η ∈ (0, 0.2]）
MAX_PAIRS = 60     # 每轮最大“听众-发言”对数（控制单次批量规模）
STATE_DIR = "data/llm"


def plan_round(w, k_events, rnd):
    reqs = []
    for _ in range(k_events):
        sp = int(w.rng.choice(w.n, p=w.speak_p))
        msg = message_text(archetype_of(w.hub, w.bridger, sp), w.b[sp],
                           TOPIC_NAMES, TOPIC_POLES)
        for u in w.g.predecessors(sp):
            reqs.append(dict(id=f"r{rnd}_u{u}_s{sp}_{len(reqs)}", listener=int(u), speaker=sp,
                             listener_archetype=archetype_of(w.hub, w.bridger, u),
                             listener_profile=verbalize(w.b[u], TOPIC_NAMES, TOPIC_POLES),
                             listener_belief=[round(float(x), 3) for x in w.b[u]],
                             edge_weight=round(float(w.g[u][sp]["weight"]), 3),
                             message=msg))
    if len(reqs) > MAX_PAIRS:
        idx = sorted(w.rng.choice(len(reqs), size=MAX_PAIRS, replace=False))
        reqs = [reqs[i] for i in idx]
    return reqs


def apply_round(w, reqs, resps, ledger, cache):
    """把归档（或实时）的 LLM 响应应用到世界状态上。

    决策落地统一走 ``llm_backend.apply_decision`` —— 与 VecWorld(backend='llm')
    和旗舰引擎的 LLM 认知层**完全同一条效应路径**，因此 Phase 2 的 LLM 臂
    与新的统一后端在语义上不可能分叉。
    """
    by_id = {r["id"]: r for r in resps}
    stats = dict(valid=0, fallback=0)
    for q in reqs:
        r = by_id.get(q["id"])
        ok = (r is not None and isinstance(r.get("new_belief"), list)
              and len(r["new_belief"]) == w.d
              and r.get("action") in ("keep", "mute", "unfollow", "block"))
        if not ok:
            stats["fallback"] += 1
            decision = LLMDecision(q["id"], "keep", None, "missing/invalid response", False)
        else:
            stats["valid"] += 1
            key = hashlib.md5((q["listener_profile"] + q["message"]
                               + str(q["listener_archetype"])).encode()).hexdigest()
            cache[key] = r
            try:
                belief = [float(x) for x in r["new_belief"]]
            except (TypeError, ValueError):
                belief = None
            decision = LLMDecision(q["id"], r["action"], belief,
                                   str(r.get("reasoning", ""))[:200], belief is not None)
        apply_decision(w, q["listener"], q["speaker"], decision, eta=ETA)
    ledger.charge(f"batch_{len(reqs)}_pairs", len(reqs))
    w.closure()
    return stats


def sdir(seed):
    d = os.path.join(STATE_DIR, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def new_world(seed, n):
    return VecWorld(n=n, backend="llm_stub", seed=seed, p_edge=0.1, p_form=0.1)


def load(seed):
    p = os.path.join(sdir(seed), "state.pkl")
    return pickle.load(open(p, "rb"))


def save(seed, st):
    pickle.dump(st, open(os.path.join(sdir(seed), "state.pkl"), "wb"))


def run_events_offline(backend, eps, seed, n, rounds, k_events):
    """相同事件结构（每轮 k_events 次发言 + 三元闭包）运行离线臂"""
    w = VecWorld(n=n, backend=backend, eps=eps, seed=seed, p_edge=0.1, p_form=0.1)
    for _ in range(rounds):
        for _ in range(k_events):
            w.step()
        w.closure()
    return w.metrics()


def replay_seed(seed, n=40, backends=("degroot", "hk", "proxy"), budget=100, eps=0.6):
    """事件级对齐重放：用归档的 LLM 请求/响应重建 Phase 2 的全部实验臂。

    为什么需要重放
    --------------
    原始 Phase 2 流程中，LLM 臂与离线臂**没有跑相同的事件序列**：
      · LLM 臂：每个种子实际执行了归档的 req_r*.json（seed0 两轮、seed1/2 各一轮）；
      · 离线臂：固定跑 ``--rounds`` 轮自生成事件（默认 6 轮）。
    直接把两者的终态指标并列，等于把「轮数差异」当成「认知后端差异」，属于无效对比。

    重放做了什么
    ------------
    1. 读取归档的 ``req_r{r}.json``（真实 LLM 运行时的事件序列，含 listener/speaker 对）；
    2. LLM 臂：按归档的 ``resp_r{r}.json`` 重放信念位移与拓扑动作，
       并使用**修正后的**动力学（含断边黑名单），因此不再受旧实现的闭包重连伪影影响；
    3. 离线臂（DeGroot / HK / Proxy）：用**同一批事件对**驱动各自的认知后端，
       从而得到事件级严格对齐的比较。

    不需要任何 API 调用：全部使用已归档的模型输出，重放过程完全确定。
    """
    d = sdir(seed)
    rounds = []
    r = 0
    while os.path.exists(os.path.join(d, f"req_r{r}.json")) and \
            os.path.exists(os.path.join(d, f"resp_r{r}.json")):
        rounds.append(r)
        r += 1
    if not rounds:
        raise FileNotFoundError(f"no archived LLM rounds for seed {seed} in {d}")

    out = {"seed": seed, "rounds_replayed": len(rounds), "arms": {}}

    # —— LLM 臂：重放归档响应 ——
    w = new_world(seed, n)
    ledger = CostLedger(budget)
    cache = {}
    valid_stats = []
    for r in rounds:
        reqs = json.load(open(os.path.join(d, f"req_r{r}.json")))
        resps = json.load(open(os.path.join(d, f"resp_r{r}.json")))
        s = apply_round(w, reqs, resps, ledger, cache)
        valid_stats.append(s)
    m = w.metrics()
    m["ledger"] = ledger.to_dict()
    m["valid_stats"] = valid_stats
    m["replayed"] = True
    out["arms"]["llm"] = m
    with open(os.path.join(d, "final_replay.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    # —— 离线臂：重放同一批事件对 ——
    event_rounds = []
    for r in rounds:
        reqs = json.load(open(os.path.join(d, f"req_r{r}.json")))
        event_rounds.append([(int(q["listener"]), int(q["speaker"])) for q in reqs])

    for be in backends:
        eps_b = eps if be == "hk" else 0.0
        wb = VecWorld(n=n, backend=be, eps=eps_b, seed=seed, p_edge=0.1, p_form=0.1)
        for pairs in event_rounds:
            wb.apply_event_pairs(pairs)
            wb.closure()
        mb = wb.metrics()
        mb["replayed"] = True
        mb["events"] = sum(len(p) for p in event_rounds)
        out["arms"][be] = mb

    return out


def cmd_replay(a):
    seeds = [int(x) for x in a.seeds.split(",")]
    rows, keys = {}, ["modularity", "num_comms", "size_gini", "small_comm_frac", "rupture_asymmetry",
                      "polarization", "giant_frac", "cross_camp_edge_frac", "edges"]
    per_arm = {}
    events_total = 0
    for s in seeds:
        rec = replay_seed(s, n=a.n, budget=a.budget)
        events_total += rec["rounds_replayed"]
        for arm, m in rec["arms"].items():
            per_arm.setdefault(arm, []).append(m)

    for arm, ms in per_arm.items():
        rows[arm] = {k: [float(m[k]) for m in ms] for k in keys}
        # 每个指标有多少个种子有定义（NaN = 该种子无互惠二元组断裂，比率未定义，不可记为 0）
        rows[arm]["_n_defined"] = {
            k: int(np.sum(np.isfinite(np.array(rows[arm][k], dtype=float)))) for k in keys}
        rows[arm]["_n_mutual_broken"] = [float(m.get("n_mutual_broken", float("nan"))) for m in ms]

    print("arm | " + " | ".join(keys) + " | n_defined(A_rupture)")
    for arm, r in rows.items():
        cells = []
        for k in keys:
            vals = np.array(r[k], dtype=float)
            finite = vals[np.isfinite(vals)]
            cells.append(f"{finite.mean():.3f}" if len(finite) else "undef")
        print(arm + " | " + " | ".join(cells)
              + f"   (n_seeds={len(r['edges'])}, A_defined={r['_n_defined']['rupture_asymmetry']}, "
                f"mutual_broken={r['_n_mutual_broken']})")
    payload = {"args": vars(a), "mode": "event-aligned replay of archived LLM responses",
               "n_rounds_total": events_total,
               "note": "所有臂重放同一批归档事件；LLM 臂使用归档模型输出，离线臂使用各自认知后端",
               "arms": rows}
    json.dump(payload, open("results/phase2_compare.json", "w"), indent=1, ensure_ascii=False)
    print("[Done] results/phase2_compare.json (event-aligned)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "apply", "finish", "compare", "replay"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    a = ap.parse_args()
    if a.cmd == "plan":
        st = load(a.seed) if os.path.exists(os.path.join(sdir(a.seed), "state.pkl")) else \
            dict(w=new_world(a.seed, a.n), ledger=CostLedger(a.budget), cache={}, rounds_done=0)
        reqs = plan_round(st["w"], a.events, a.round)
        st["pending"] = reqs
        save(a.seed, st)
        json.dump(reqs, open(os.path.join(sdir(a.seed), f"req_r{a.round}.json"), "w"), indent=1)
        print(f"planned {len(reqs)} pairs -> {sdir(a.seed)}/req_r{a.round}.json")
    elif a.cmd == "apply":
        st = load(a.seed)
        resps = json.load(open(os.path.join(sdir(a.seed), f"resp_r{a.round}.json")))
        s = apply_round(st["w"], st["pending"], resps, st["ledger"], st["cache"])
        st["rounds_done"] = a.round + 1
        st.setdefault("valid_stats", []).append(s)
        save(a.seed, st)
        print("applied", s, "calls used", st["ledger"].calls, "edges", st["w"].g.number_of_edges())
    elif a.cmd == "finish":
        st = load(a.seed)
        m = st["w"].metrics()
        m["ledger"] = st["ledger"].to_dict(); m["valid_stats"] = st.get("valid_stats")
        json.dump(m, open(os.path.join(sdir(a.seed), "final.json"), "w"), indent=2)
        print(json.dumps({k: v for k, v in m.items() if k != "ledger"}, indent=1))
    elif a.cmd == "replay":
        cmd_replay(a)
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
        rows, keys = {}, ["modularity", "num_comms", "size_gini", "small_comm_frac", "rupture_asymmetry",
                          "polarization", "giant_frac", "cross_camp_edge_frac", "edges"]
        for name, be, eps in (("llm", None, 0), ("degroot", "degroot", 0), ("hk0.6", "hk", 0.6),
                              ("hk0.9", "hk", 0.9), ("proxy", "proxy", 0)):
            per = []
            for s in seeds:
                if be is None:
                    p = os.path.join(sdir(s), "final.json")
                    if os.path.exists(p):
                        per.append(json.load(open(p)))
                else:
                    per.append(run_events_offline(be, eps, s, a.n, a.rounds, a.events))
            rows[name] = {k: [float(r[k]) for r in per] for k in keys}
        print("arm | " + " | ".join(keys))
        for name, r in rows.items():
            print(name + " | " + " | ".join(f"{np.nanmean(r[k]):.3f}" if len(r[k]) else "-" for k in keys)
                  + f"   (n_seeds={len(r['edges'])})")
        json.dump(rows, open("results/phase2_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
