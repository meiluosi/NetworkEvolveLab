"""Phase 8: external-validity check on a public, human-labeled stance graph.

Dataset: Political Blogosphere (Adamic & Glance 2005) -- 1,490 blogs with
left/right ``value`` labels and directed blog-to-blog links.  We induce a
100-node subgraph per seed (top-degree pool, seeded sample), keep the original
stance labels as belief priors, and compare the flagship engine against three
static baselines.  This is a real-stance complement to the SNAP Higgs subgraph,
which has no stance labels.

Usage:
    python -m src.network_evolve.phase8_real_stance --seeds 10 --rounds 25 \
        --nodes 100 --out results/phase8_real_stance.json
"""
import argparse
import json

import numpy as np

from .benchmark_suite import instantiate_model, run_engine_method
from .datasets_stance import load_polblogs_real_subgraph, polblogs_stats
from .expkit import fmt

METHODS = ["DeGroot", "Hegselmann_Krause", "Friedkin_Johnsen",
           "Ours_CoEvolution_Engine"]
METRICS = ["modularity_q", "polarization", "giant_frac", "small_comm_frac",
           "rupture_asymmetry", "num_comms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--out", default="results/phase8_real_stance.json")
    a = ap.parse_args()

    log = []

    def P(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        log.append(line)

    P(f"=== Phase 8: PolBlogs real-stance external validity "
      f"(seeds={a.seeds}, rounds={a.rounds}, nodes={a.nodes}) ===")
    stats = [polblogs_stats(a.nodes, s) for s in range(a.seeds)]
    P("graph stats: " + json.dumps({
        "num_nodes": stats[0]["num_nodes"],
        "mean_edges": float(np.mean([s["num_edges"] for s in stats])),
        "mean_reciprocity": float(np.mean([s["reciprocity"] for s in stats])),
        "mean_Q0": float(np.mean([s["modularity_q"] for s in stats])),
        "mean_left_frac": float(np.mean([s["left_frac"] for s in stats])),
    }, ensure_ascii=False))

    out = {"args": vars(a), "dataset": "PolBlogs_Real", "graph_stats": stats,
           "benchmark": {}}
    P("\nmethod | Q | polarization | giant | small-comm | A_rupture")
    for method in METHODS:
        runs = []
        for s in range(a.seeds):
            g, ideo = load_polblogs_real_subgraph(a.nodes, s)
            if method == "Ours_CoEvolution_Engine":
                res = run_engine_method(g, ideo, s, a.rounds)
            else:
                res = instantiate_model(method, g, ideo, s).run(rounds=a.rounds)
            runs.append(res)
        row = {k: fmt([r[k] for r in runs if not np.isnan(r[k])]) for k in METRICS}
        out["benchmark"][method] = row
        P(f"{method:<24} | {row['modularity_q']} | {row['polarization']} | "
          f"{row['giant_frac']} | {row['small_comm_frac']} | {row['rupture_asymmetry']}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(a.out.replace(".json", ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"[Done] {a.out}")


if __name__ == "__main__":
    main()
