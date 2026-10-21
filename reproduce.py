#!/usr/bin/env python3
"""One-click reproduction CLI for NetworkEvolveLab experiments.

Usage:
    python reproduce.py --all           # Run all benchmark evaluations and generate figures
    python reproduce.py --benchmarks    # Run 6 SOTA baselines across 3 real datasets
    python reproduce.py --figures       # Re-generate Figure 1 to Figure 6
    python reproduce.py --test          # Run complete test suite (9/9 PASS)
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
    parser.add_argument("--figures", action="store_true", help="Render publication Figures 1-6 (PNG + SVG)")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--nodes", type=int, default=60, help="Number of nodes per empirical graph")
    parser.add_argument("--rounds", type=int, default=25, help="Simulation rounds")
    parser.add_argument("--seeds", type=int, default=10, help="Random seeds for 95% CI")
    
    args = parser.parse_args()

    if not any([args.all, args.benchmarks, args.figures, args.test]):
        parser.print_help()
        sys.exit(0)

    if args.test or args.all:
        run_cmd(f"{sys.executable} -m unittest discover tests -v", "Running 9 Unit Tests")

    if args.benchmarks or args.all:
        cmd = f"{sys.executable} -m src.network_evolve.benchmark_suite --nodes {args.nodes} --rounds {args.rounds} --seeds {args.seeds} --out results/sota_benchmark.json"
        run_cmd(cmd, "Running 6 SOTA Baselines across 3 Empirical Datasets")

    if args.figures or args.all:
        run_cmd(f"{sys.executable} scripts/generate_figures.py", "Generating Publication Figures 1 to 6 (results/figures/)")

    print("\n========================================================")
    print(" [✓] Reproduction pipeline finished successfully!")
    print("========================================================")

if __name__ == "__main__":
    main()
