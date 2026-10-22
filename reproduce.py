#!/usr/bin/env python3
"""One-click reproduction CLI for NetworkEvolveLab experiments.

Usage:
    python reproduce.py --all           # Run all benchmark evaluations and generate figures
    python reproduce.py --benchmarks    # Run the unified SOTA benchmark suite (6 datasets x 7 methods)
    python reproduce.py --experiments   # Run every phase experiment (0, 1, 1b, 1c, 2, 3, 4, 5, 7)
    python reproduce.py --llm-arm       # Phase 6: real-LLM cognitive layer vs rule backend (needs LLM credentials)
    python reproduce.py --figures       # Re-generate Figure 1 to Figure 7 + auto-generated paper tables
    python reproduce.py --paper         # Build the ICWSM single-file submission and compile the PDF
    python reproduce.py --provenance    # Record / verify the experiment provenance manifest
    python reproduce.py --test          # Run complete test suite
"""
import sys
import argparse
import subprocess

def run_cmd(cmd: str, desc: str):
    print(f"\n[+] {desc}...")
    print(f"    Command: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[-] FAILED with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✓] {desc} succeeded.")

def main():
    parser = argparse.ArgumentParser(description="NetworkEvolveLab Reproduction Harness")
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite and figure generation")
    parser.add_argument("--benchmarks", action="store_true", help="Run multi-method multi-dataset benchmark suite")
    parser.add_argument("--experiments", action="store_true", help="Run all phase experiments (0/1/1b/2/3/4/5)")
    parser.add_argument("--figures", action="store_true", help="Render publication Figures 1-7 (PNG + SVG)")
    parser.add_argument("--paper", action="store_true",
                        help="Build the ICWSM single-file submission package and compile the PDF")
    parser.add_argument("--llm-arm", action="store_true",
                        help="Phase 6: run the real-LLM cognitive layer experiment (needs LLM credentials)")
    parser.add_argument("--provenance", action="store_true", help="Record / verify experiment provenance manifest")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--nodes", type=int, default=60, help="Number of nodes per empirical graph")
    parser.add_argument("--rounds", type=int, default=25, help="Simulation rounds")
    parser.add_argument("--seeds", type=int, default=10, help="Random seeds for 95% CI")

    args = parser.parse_args()

    if not any([args.all, args.benchmarks, args.experiments, args.figures, args.paper,
                args.llm_arm, args.provenance, args.test]):
        parser.print_help()
        sys.exit(0)

    if args.test or args.all:
        run_cmd(f"{sys.executable} -m unittest discover tests -v", "Running full unit + invariant test suite")

    if args.experiments or args.all:
        py = sys.executable
        run_cmd(f"{py} -m src.network_evolve.phase0 --nodes 100 --rounds 20 --seeds 5 --out results/phase0_sweep.json",
                "Phase 0: dynamics baseline & phase-transition sweep")
        run_cmd(f"{py} -m src.network_evolve.phase1 --nodes 100 --rounds 20 --seeds 10 --theta 0.4 --block 1.2 --out results/phase1.json",
                "Phase 1: homogeneous vs heterogeneous archetype A/B")
        run_cmd(f"{py} -m src.network_evolve.phase1b --seeds 20 --rounds 60 --out results/phase1b.json",
                "Phase 1b: H2 ablation + H1 decision grid")
        run_cmd(f"{py} -m src.network_evolve.phase_robustness --seeds 10 --rounds 40 --out results/robustness_sweep.json",
                "Cross-topology robustness sweep (H2 + intervention by topology)")
        run_cmd(f"{py} -m src.network_evolve.phase1c_threshold --seeds 10 --rounds 40 --nodes 100 --out results/phase1c_threshold.json",
                "Phase 1c: cross-topology critical-threshold measurement (fine theta grid)")
        run_cmd(f"{py} -m src.network_evolve.phase2_offline --seeds 20 --n 60 --rounds 30 --out results/phase2_offline.json",
                "Phase 2: offline cognitive backends (DeGroot / HK / Proxy)")
        run_cmd(f"{py} -m src.network_evolve.phase3 --seeds 20 --rounds 60 --out results/phase3.json",
                "Phase 3: counterfactual platform interventions (H3)")
        run_cmd(f"{py} -m src.network_evolve.phase4_attribution --seeds 20 --rounds 30 --nodes 60 --out results/phase4_attribution.json",
                "Phase 4: asymmetric-rupture attribution ablation")

    if args.benchmarks or args.all:
        cmd = (f"{sys.executable} -m src.network_evolve.benchmark_suite --nodes {args.nodes} "
               f"--rounds {args.rounds} --seeds {args.seeds} --out results/sota_benchmark.json")
        run_cmd(cmd, "Running unified SOTA benchmark suite (datasets x methods)")

    if args.experiments or args.all:
        run_cmd(f"{sys.executable} -m src.network_evolve.phase5_bound_audit --seeds 10 --rounds 60 --out results/phase5_bound_audit.json",
                "Phase 5: generalized modularity-bound consistency audit")
        run_cmd(f"{sys.executable} -m src.network_evolve.phase7_scaling --sizes 60,100,250,500,1000 "
                "--seeds 10 --rounds 60 --real-higgs --out results/phase7_scaling.json",
                "Phase 7: network-size sensitivity study (N up to 1000, plus real Higgs subgraph)")

    if args.llm_arm:
        run_cmd(f"{sys.executable} -m src.network_evolve.phase6_llm_engine --seeds 10 --rounds 30 "
                "--nodes 60 --llm-events 6 --budget-calls 400 --out results/phase6_llm_engine.json",
                "Phase 6: real-LLM cognitive layer vs rule backend (paired, seed-level)")

    if args.figures or args.all:
        run_cmd(f"{sys.executable} scripts/generate_figures.py", "Generating Publication Figures 1 to 7 (results/figures/)")
        run_cmd(f"{sys.executable} scripts/generate_paper_tables.py", "Regenerating auto-generated LaTeX tables (paper/tables/)")
        run_cmd(f"{sys.executable} scripts/check_paper_integrity.py", "Checking paper LaTeX static integrity")

    if args.paper or args.all:
        run_cmd(f"{sys.executable} scripts/build_submission.py", "Building ICWSM single-file submission package")
        run_cmd(f"{sys.executable} scripts/build_submission.py --check", "Verifying submission package is in sync with sources")
        run_cmd("tectonic -X compile build/icwsm/NetworkEvolveLab_ICWSM.tex --outdir build/icwsm",
                "Compiling the paper PDF (requires tectonic on PATH)")

    if args.provenance or args.all:
        run_cmd(f"{sys.executable} scripts/record_provenance.py", "Recording experiment provenance manifest")

    print("\n========================================================")
    print(" [✓] Reproduction pipeline finished successfully!")
    print("========================================================")

if __name__ == "__main__":
    main()
