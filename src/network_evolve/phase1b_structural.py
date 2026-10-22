"""Phase 1b-structural: Bridger 原型按**结构位置**选择的对照臂。

原 H2 中 Bridger 是按节点索引分配的（`i < num_hubs + num_bridgers`），
因此「Bridger 更抗断裂」可能只是硬编码容忍度 0.7 的作用，而不是结构洞位置。
本脚本保持原型比例与所有参数不变，只把 Hub 按度、Bridger 按 betweenness
重新分配，重跑 H2 的 base 与等容忍度两行，用于检验结构位置是否改变结论。

用法：
    python -m src.network_evolve.phase1b_structural --seeds 20 --rounds 60 \
        --out results/phase1b_structural.json
"""
import argparse
import json

from .expkit import run_many, h2_summary, fmt

log_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--out", default="results/phase1b_structural.json")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    out = {"args": vars(a), "arms": {}, "design": {
        "assignment": "hub=highest degree; bridger=highest betweenness among non-hubs; "
                      "long-tail=rest",
        "proportions": "2% hub / 8.5% bridger / 89.5% long-tail (unchanged)",
        "purpose": "test whether the H2 retention advantage depends on structural position "
                   "rather than on the assigned archetype label",
    }}

    configs = {
        "index_base": dict(),
        "index_tol_equal": dict(bridger_dtheta=0.0),
        "structural_base": dict(bridger_assignment="structural"),
        "structural_tol_equal": dict(bridger_assignment="structural", bridger_dtheta=0.0),
    }
    P(f"## H2 structural-position control (T={a.rounds}, seeds={a.seeds})")
    P("arm | bridger_ret | other_ret | gap(rel) [CI95] | p(seed-level) | H2")
    for name, cfg in configs.items():
        runs = run_many({**cfg, "block": 1.2, "rounds": a.rounds}, seeds)
        h = h2_summary(runs)
        out["arms"][name] = h
        P(f"{name:<24} | {h['bridger_mean']:.3f} | {h['other_mean']:.3f} | "
          f"{h['relative_gap']:+.3f} [{h['gap_ci95'][0]:+.3f},{h['gap_ci95'][1]:+.3f}] | "
          f"{h['p_value']:.3g} | {h['H2_supported']}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(a.out.replace(".json", ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[Done] {a.out}")


if __name__ == "__main__":
    main()
