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

TOPICS = [
    ("binding government regulation of large AI models", "oppose", "support"),
    ("trading user privacy for faster innovation", "support", "oppose"),
    ("open-sourcing frontier model weights", "support", "oppose"),
]
ETA = 0.2          # 阻尼学习率（ARCHITECTURE.md: η ∈ (0, 0.2]）
MAX_PAIRS = 60     # 每轮最大“听众-发言”对数（控制单次批量规模）
STATE_DIR = "data/llm"


def verbalize(b, first_person=False):
    def word(x):
        a = abs(x)
        return "strongly" if a > 0.6 else "moderately" if a > 0.3 else "weakly"
    parts = []
    for k, (topic, neg, pos) in enumerate(TOPICS):
        x = float(b[k])
        verb = (pos if x > 0 else neg)
        if not first_person:
            verb = verb + "s"                      # supports / opposes
        if abs(x) <= 0.1:
            parts.append(f"{'am' if first_person else 'is'} neutral about {topic}")
        else:
            parts.append(f"{word(x)} {verb} {topic}")
    return "; ".join(parts)


def message_text(archetype, b):
    style = {"hub": "Announcing loudly: ", "bridger": "Trying to see both sides: ", "tail": "Just my two cents: "}[archetype]
    return style + "I " + verbalize(b, first_person=True) + "."


def archetype_of(w, i):
    return "hub" if i in w.hub else "bridger" if i in w.bridger else "tail"


def plan_round(w, k_events, rnd):
    reqs = []
    for _ in range(k_events):
        sp = int(w.rng.choice(w.n, p=w.speak_p))
        msg = message_text(archetype_of(w, sp), w.b[sp])
        for u in w.g.predecessors(sp):
            reqs.append(dict(id=f"r{rnd}_u{u}_s{sp}_{len(reqs)}", listener=int(u), speaker=sp,
                             listener_archetype=archetype_of(w, u),
                             listener_profile=verbalize(w.b[u]),
                             listener_belief=[round(float(x), 3) for x in w.b[u]],
                             edge_weight=round(float(w.g[u][sp]["weight"]), 3),
                             message=msg))
    if len(reqs) > MAX_PAIRS:
        idx = sorted(w.rng.choice(len(reqs), size=MAX_PAIRS, replace=False))
        reqs = [reqs[i] for i in idx]
    return reqs


def apply_round(w, reqs, resps, ledger, cache):
    by_id = {r["id"]: r for r in resps}
    stats = dict(valid=0, fallback=0)
    for q in reqs:
        r = by_id.get(q["id"])
        ok = (r is not None and isinstance(r.get("new_belief"), list) and len(r["new_belief"]) == w.d
              and r.get("action") in ("keep", "mute", "unfollow", "block"))
        key = hashlib.md5((q["listener_profile"] + q["message"] + str(q["listener_archetype"])).encode()).hexdigest()
        if not ok:
            stats["fallback"] += 1
            r = dict(new_belief=[float(x) for x in w.b[q["listener"]]], action="keep")  # 缺失响应：保持不变，并记录
        else:
            stats["valid"] += 1
            cache[key] = r
        u, sp = q["listener"], q["speaker"]
        delta = np.clip(np.array(r["new_belief"], dtype=float) - w.b[u], -0.5, 0.5)
        w.b[u] = np.clip(w.b[u] + ETA * delta, -1, 1)           # 阻尼更新
        if not w.g.has_edge(u, sp):
            continue
        if r["action"] == "keep":
            w.g[u][sp]["weight"] = min(1.0, w.g[u][sp]["weight"] + 0.05)
        elif r["action"] == "mute":
            nw = w.g[u][sp]["weight"] - 0.15
            if nw <= 0:
                w.g.remove_edge(u, sp); w.broken_edges.append((u, sp))
            else:
                w.g[u][sp]["weight"] = nw
        else:  # unfollow / block
            w.g.remove_edge(u, sp); w.broken_edges.append((u, sp))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "apply", "finish", "compare"])
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
