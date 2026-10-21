"""Phase 0：离线动力学基线 —— 扫描容忍度 θ 寻找拓扑相变临界点（无需 API）。
用法: python -m src.network_evolve.phase0 --nodes 100 --rounds 20 --seeds 5 --out results/phase0_sweep.json
"""
import argparse
import json
import numpy as np
from .engine import CoEvolutionEngine
from .metrics import snapshot


def run_one(theta, seed, nodes, rounds, p_form):
    e = CoEvolutionEngine(num_nodes=nodes, seed=seed, base_tolerance=theta,
                          bridger_tolerance=min(1.0, theta + 0.3), p_form=p_form)
    init = snapshot(e)
    for _ in range(rounds):
        e.run_round()
    return init, snapshot(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--p_form", type=float, default=0.1)
    ap.add_argument("--out", default="results/phase0_sweep.json")
    a = ap.parse_args()
    thetas = [round(x, 2) for x in np.arange(0.2, 0.95, 0.05)]
    rows = []
    print("theta |  Q_init  Q_end  |  Pi_end | cross | giant | comps | edges")
    for th in thetas:
        runs = [run_one(th, s, a.nodes, a.rounds, a.p_form) for s in range(a.seeds)]
        agg = {k: float(np.mean([r[1][k] for r in runs])) for k in runs[0][1]}
        agg["modularity_q_init"] = float(np.mean([r[0]["modularity_q"] for r in runs]))
        agg["modularity_q_std"] = float(np.std([r[1]["modularity_q"] for r in runs]))
        agg["theta"] = th
        rows.append(agg)
        print(f"{th:.2f}  | {agg['modularity_q_init']:.3f} {agg['modularity_q']:.3f} | "
              f"{agg['polarization']:.3f} | {agg['cross_edge_frac']:.2f} | {agg['giant_frac']:.2f} | "
              f"{agg['num_components']:.1f} | {agg['num_edges']:.0f}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
