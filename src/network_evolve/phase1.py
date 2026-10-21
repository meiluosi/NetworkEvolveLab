"""Phase 1：A/B 对照（同质 vs 异质原型）、Bridger 留存(H2)、拉黑 + 推荐摩擦力扫描(RQ2)。
用法: python -m src.network_evolve.phase1 --out results/phase1.json
"""
import argparse
import json
import numpy as np
from .engine import CoEvolutionEngine
from .metrics import snapshot, init_incident_edges, retention_by_node, h2_test


def run(seed, nodes, rounds, theta, hetero, block, friction, p_form=0.1):
    e = CoEvolutionEngine(num_nodes=nodes, seed=seed, base_tolerance=theta,
                          bridger_tolerance=min(1.0, theta + 0.3), p_form=p_form,
                          heterogeneous=hetero, block_diff=block, rec_friction=friction)
    init = init_incident_edges(e)
    for _ in range(rounds):
        e.run_round()
    snap = snapshot(e)
    snap["num_blocks"] = len(e.blocked)
    return e, snap, retention_by_node(e, init)


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--block", type=float, default=1.2)
    ap.add_argument("--out", default="results/phase1.json")
    a = ap.parse_args()
    out = {"args": vars(a)}
    keys = ["modularity_louvain", "modularity_q", "num_louvain_comms", "within_comm_belief_var",
            "polarization", "cross_edge_frac", "giant_frac", "isolated_nodes", "num_big_components", "num_edges"]

    print("== A/B 对照 (theta=%.2f, block=%.1f, friction=1.0) ==" % (a.theta, a.block))
    out["ab"] = {}
    for name, hetero in (("A_homogeneous", False), ("B_heterogeneous", True)):
        rs = [run(s, a.nodes, a.rounds, a.theta, hetero, a.block, 1.0) for s in range(a.seeds)]
        rows = [r[1] for r in rs]
        agg = {k: mean(rows, k) for k in keys + ["num_blocks"]}
        br, ot = [], []
        for e, _, ret in rs:
            for i, v in ret.items():
                (br if e.agents[i].archetype == "active_bridger" else ot).append(v)
        agg["H2"] = h2_test(br, ot)
        out["ab"][name] = agg
        print(name, {k: round(v, 3) for k, v in agg.items() if k != "H2"})
        print("   H2:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg["H2"].items()})

    print("== RQ2: 推荐摩擦力 rho x 拉黑 (异质组) ==")
    print("block  rho | Q_louv comms within  | cross | giant isolated bigcomp blocks")
    out["rq2"] = []
    for block in (None, a.block):
        for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows = [run(s, a.nodes, a.rounds, a.theta, True, block, rho)[1] for s in range(a.seeds)]
            agg = {k: mean(rows, k) for k in keys + ["num_blocks"]}
            agg.update(block=block, rho=rho)
            out["rq2"].append(agg)
            print(f"{str(block):5s} {rho:.2f} | {agg['modularity_louvain']:.3f} {agg['num_louvain_comms']:.1f} "
                  f"{agg['within_comm_belief_var']:.3f} | {agg['cross_edge_frac']:.2f} | {agg['giant_frac']:.2f} "
                  f"{agg['isolated_nodes']:.1f} {agg['num_big_components']:.1f} {agg['num_blocks']:.0f}")
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
