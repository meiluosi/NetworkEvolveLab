"""Phase 2 离线对照：DeGroot vs HK vs 语义规则代理(proxy)。proxy 不是 LLM——仅作为预注册的“类 LLM 假设”，
真实 LLM 臂见 phase2_llm.py。用法: python -m src.network_evolve.phase2_offline --seeds 20"""
import argparse, json
import numpy as np
from .semantic import VecWorld
from .expkit import mean_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default="results/phase2_offline.json")
    a = ap.parse_args()
    out, lines = {}, []
    keys = ["modularity", "num_comms", "size_gini", "small_comm_frac", "rupture_asymmetry",
            "polarization", "giant_frac", "cross_camp_edge_frac", "edges"]
    hdr = "backend | " + " | ".join(keys)
    print(hdr); lines.append(hdr)
    for be, eps in (("degroot", 0), ("hk", 0.6), ("hk", 0.9), ("hk", 1.2), ("proxy", 0)):
        rs = [VecWorld(n=a.n, backend=be, eps=eps, seed=s).run(a.rounds) for s in range(a.seeds)]
        row = {}
        for k in keys:
            vals = [r[k] for r in rs if not np.isnan(r[k])]
            row[k] = mean_ci(vals) if vals else (float("nan"), 0.0)
        name = be if be != "hk" else f"hk(eps={eps})"
        out[name] = row
        line = f"{name} | " + " | ".join(f"{row[k][0]:.3f}±{row[k][1]:.3f}" for k in keys)
        print(line); lines.append(line)
    json.dump(out, open(a.out, "w"), indent=2)
    open(a.out.replace(".json", ".log"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
