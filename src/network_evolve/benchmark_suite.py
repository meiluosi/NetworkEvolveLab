"""综合基准与多 SOTA 方法评测套件（Unified Benchmark & SOTA Suite）：
跨三大黄金实证数据集（Twitter Election, SNAP Higgs, Reddit ConvoKit），
评测 6 种观点动力学与拓扑演化方法：
  1. DeGroot (1974)
  2. Hegselmann-Krause (2002)
  3. Friedkin-Johnsen (1990)
  4. Generative Agents (Park et al., 2023)
  5. Prompt-Rewire (Gu et al., 2025)
  6. Ours (NetworkEvolveLab Co-evolution)
"""
import argparse
import json
import numpy as np
from typing import Dict, List, Any
from .datasets import load_twitter_election, load_snap_higgs, load_reddit_convokit
from .baselines import (
    DeGrootModel, HegselmannKrauseModel, FriedkinJohnsenModel,
    GenerativeAgentsMemoryModel, GuEtAlPromptRewireModel, OursCoEvolutionModel
)
from .expkit import mean_ci, fmt

DATASETS = ["Twitter_Election_2020", "SNAP_Higgs_Twitter", "Reddit_ConvoKit"]
MODELS = ["DeGroot", "Hegselmann_Krause", "Friedkin_Johnsen", "Park_et_al_2023", "Gu_et_al_2025", "Ours_CoEvolution"]

def get_dataset(dname: str, n: int, s: int):
    if dname == "Twitter_Election_2020":
        return load_twitter_election(num_nodes=n, seed=s)
    elif dname == "SNAP_Higgs_Twitter":
        g = load_snap_higgs(num_nodes=n, seed=s)
        rng = np.random.default_rng(s)
        # Higgs 双极化先验
        ideo = {i: float(rng.choice([-0.7, 0.7]) + rng.normal(0, 0.15)) for i in range(n)}
        return g, ideo
    elif dname == "Reddit_ConvoKit":
        g, topics = load_reddit_convokit(num_nodes=n, seed=s)
        ideo = {i: topics[i][0] for i in range(n)}  # 主议题
        return g, ideo
    raise ValueError(dname)

def instantiate_model(mname: str, g, ideo, s: int):
    if mname == "DeGroot":
        return DeGrootModel(g, ideo, seed=s)
    elif mname == "Hegselmann_Krause":
        return HegselmannKrauseModel(g, ideo, epsilon=0.45, seed=s)
    elif mname == "Friedkin_Johnsen":
        return FriedkinJohnsenModel(g, ideo, stubbornness=0.35, seed=s)
    elif mname == "Park_et_al_2023":
        return GenerativeAgentsMemoryModel(g, ideo, memory_window=5, seed=s)
    elif mname == "Gu_et_al_2025":
        return GuEtAlPromptRewireModel(g, ideo, seed=s)
    elif mname == "Ours_CoEvolution":
        return OursCoEvolutionModel(g, ideo, theta=0.40, block_diff=1.2, div_rec=0.15, seed=s)
    raise ValueError(mname)

def main():
    parser = argparse.ArgumentParser(description="Unified SOTA Benchmark Evaluation")
    parser.add_argument("--nodes", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/sota_benchmark.json")
    args = parser.parse_args()

    print(f"=== SOTA 方法与黄金数据集大一统综合评测 (Seeds={args.seeds}, Nodes={args.nodes}, Rounds={args.rounds}) ===")
    
    results = {"args": vars(args), "benchmark": {}}
    metric_keys = ["modularity_q", "polarization", "giant_frac", "small_comm_frac", "rupture_asymmetry"]

    for dname in DATASETS:
        print(f"\n>>> Dataset: {dname}")
        results["benchmark"][dname] = {}
        print("Model Name              | Modularity Q   | Polarization Π | Giant Comp S   | Small Comm %   | Asym Rupture A")
        print("-" * 105)
        for mname in MODELS:
            runs = []
            for s in range(args.seeds):
                g, ideo = get_dataset(dname, args.nodes, s)
                model = instantiate_model(mname, g, ideo, s)
                res = model.run(rounds=args.rounds)
                runs.append(res)
            
            row = {}
            for k in metric_keys:
                vals = [r[k] for r in runs if not np.isnan(r[k])]
                row[k] = fmt(vals) if vals else "N/A"
            
            results["benchmark"][dname][mname] = row
            print(f"{mname:<23} | {row['modularity_q']:<14} | {row['polarization']:<14} | {row['giant_frac']:<14} | {row['small_comm_frac']:<14} | {row['rupture_asymmetry']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Benchmark results successfully written to {args.out}")

if __name__ == "__main__":
    main()
