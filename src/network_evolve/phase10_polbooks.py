"""Phase 10: external-validity check on the human-labeled PolBooks graph."""
import argparse
import json

import numpy as np

from .benchmark_suite import instantiate_model, run_engine_method
from .datasets_polbooks import load_polbooks_real_subgraph
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
    ap.add_argument("--out", default="results/phase10_polbooks.json")
    a = ap.parse_args()
    log = []

    def P(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True); log.append(line)

    P(f"=== Phase 10: PolBooks human-labeled external validity "
      f"(seeds={a.seeds}, rounds={a.rounds}, nodes={a.nodes}) ===")
    stats = []
    for s in range(a.seeds):
        g, ideo = load_polbooks_real_subgraph(a.nodes, s)
        v = np.array(list(ideo.values()))
        stats.append({"num_nodes": g.number_of_nodes(), "num_edges": g.number_of_edges(),
                      "mean_label": float(v.mean()), "neg_frac": float((v < 0).mean())})
    P("graph stats: " + json.dumps({
        "mean_edges": float(np.mean([x["num_edges"] for x in stats])),
        "mean_label": float(np.mean([x["mean_label"] for x in stats])),
        "mean_neg_frac": float(np.mean([x["neg_frac"] for x in stats])),
    }, ensure_ascii=False))
    out = {"args": vars(a), "dataset": "PolBooks_Real", "graph_stats": stats,
           "benchmark": {}}
    for method in METHODS:
        runs = []
        for s in range(a.seeds):
            g, ideo = load_polbooks_real_subgraph(a.nodes, s)
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
