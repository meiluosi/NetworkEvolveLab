"""实验结果溯源清单（Experiment Provenance Manifest）。

为 results/ 下每一个冻结的实验产物记录：生成命令、参数、文件 SHA256、字节数、
生成时间、代码库 Git 提交、以及运行时环境（Python / 依赖版本）。

用途（对应 ARS 完整性门的 Mode 1/3/6 检测项）：
  · Mode 3（幻觉实验结果）—— 每个数字都能追溯到「哪条命令、哪个种子集合、哪次提交」；
  · Mode 6（方法学捏造）—— 论文引用的实验参数与实际运行配置一一对应；
  · Mode 1（实现缺陷）—— 结果文件与代码提交绑定，代码变更后必须重跑并更新清单。

用法：
  python scripts/record_provenance.py            # 写入/更新 results/PROVENANCE.json
  python scripts/record_provenance.py --check    # CI 模式：结果文件与清单不一致则退出码 1
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MANIFEST = os.path.join(RESULTS, "PROVENANCE.json")

# 结果文件 -> 生成命令（与实际执行保持一致）
COMMANDS = {
    "phase0_sweep.json": "python -m src.network_evolve.phase0 --nodes 100 --rounds 20 --seeds 5 --out results/phase0_sweep.json",
    "phase1.json": "python -m src.network_evolve.phase1 --nodes 100 --rounds 20 --seeds 10 --theta 0.4 --block 1.2 --out results/phase1.json",
    "phase1b.json": "python -m src.network_evolve.phase1b --seeds 20 --rounds 60 --out results/phase1b.json",
    "phase1b_structural.json": "python -m src.network_evolve.phase1b_structural --seeds 20 --rounds 60 --out results/phase1b_structural.json",
    "phase1c_threshold.json": "python -m src.network_evolve.phase1c_threshold --seeds 10 --rounds 40 --nodes 100 --theta-min 0.15 --theta-max 0.85 --out results/phase1c_threshold.json",
    "phase1c_matched.json": "python -m src.network_evolve.phase1c_threshold --seeds 10 --rounds 40 --nodes 100 --theta-min 0.15 --theta-max 0.85 --matched-degree 6.6 --matched-reciprocity 0.65 --out results/phase1c_matched.json",
    "phase8_real_stance.json": "python -m src.network_evolve.phase8_real_stance --seeds 10 --rounds 25 --nodes 100 --out results/phase8_real_stance.json",
    "phase2_offline.json": "python -m src.network_evolve.phase2_offline --seeds 20 --n 60 --rounds 30 --out results/phase2_offline.json",
    "phase2_compare.json": "python -m src.network_evolve.phase2_llm replay --seeds 0,1,2 --n 40",
    "phase3.json": "python -m src.network_evolve.phase3 --seeds 20 --rounds 60 --out results/phase3.json",
    "phase4_attribution.json": "python -m src.network_evolve.phase4_attribution --seeds 20 --rounds 30 --nodes 60 --out results/phase4_attribution.json",
    "phase5_bound_audit.json": "python -m src.network_evolve.phase5_bound_audit --seeds 10 --rounds 60 --out results/phase5_bound_audit.json",
    "sota_benchmark.json": "python -m src.network_evolve.benchmark_suite --nodes 60 --rounds 25 --seeds 10 --out results/sota_benchmark.json",
    "robustness_sweep.json": "python -m src.network_evolve.phase_robustness --seeds 10 --rounds 40 --out results/robustness_sweep.json",
    "phase6_llm_engine.json": "python -m src.network_evolve.phase6_llm_engine --seeds 10 --rounds 30 --nodes 60 --llm-events 6 --budget-calls 400 --out results/phase6_llm_engine.json",
    "phase7_scaling.json": "python -m src.network_evolve.phase7_scaling --sizes 60,100,250,500,1000 --seeds 10 --rounds 60 --fixed-p-large-sizes 1000 --fixed-p-large-seeds 3 --convergence-check --real-higgs --out results/phase7_scaling.json",
    # .log 是对应命令 stdout 的留档（stdout 重定向到该文件），因此共用同一条命令
    "phase1b.log": "python -m src.network_evolve.phase1b --seeds 20 --rounds 60 --out results/phase1b.json  # stdout 留档",
    "phase1b_structural.log": "python -m src.network_evolve.phase1b_structural --seeds 20 --rounds 60 --out results/phase1b_structural.json  # stdout 留档",
    "phase8_real_stance.log": "python -m src.network_evolve.phase8_real_stance --seeds 10 --rounds 25 --nodes 100 --out results/phase8_real_stance.json  # stdout 留档",
    "phase2_offline.log": "python -m src.network_evolve.phase2_offline --seeds 20 --n 60 --rounds 30 --out results/phase2_offline.json  # stdout 留档",
    "phase4_attribution.log": "python -m src.network_evolve.phase4_attribution --seeds 20 --rounds 30 --nodes 60 --out results/phase4_attribution.json  # stdout 留档",
}
LLM_ARTIFACT_COMMAND = (
    "python -m src.network_evolve.phase2_llm plan|apply|finish --seed S --round R "
    "(外部 LLM 后端读取 req_r{R}.json 并写出 resp_r{R}.json；ledger 记录批量调用次数)"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_artifacts():
    """收集需要纳管的结果产物。"""
    items = []
    for fname in sorted(os.listdir(RESULTS)):
        if not fname.endswith((".json", ".log")):
            continue
        if fname == "PROVENANCE.json":
            continue
        items.append(("results/" + fname, COMMANDS.get(fname)))
    llm_dir = os.path.join(ROOT, "data", "llm")
    if os.path.isdir(llm_dir):
        for seed_dir in sorted(os.listdir(llm_dir)):
            sd = os.path.join(llm_dir, seed_dir)
            if not os.path.isdir(sd):
                continue
            for fname in sorted(os.listdir(sd)):
                if fname.endswith(".json"):
                    items.append((f"data/llm/{seed_dir}/{fname}", LLM_ARTIFACT_COMMAND))
    bench_dir = os.path.join(ROOT, "data", "benchmarks")
    if os.path.isdir(bench_dir):
        for fname in sorted(os.listdir(bench_dir)):
            if fname.endswith(".json"):
                if fname.startswith("polblogs_"):
                    cmd = ("python scripts/build_polblogs_caches.py --nodes 100 --seeds 0-9 "
                           "# PolBlogs human-labeled stance subgraph cache")
                else:
                    cmd = ("src/network_evolve/benchmark_suite.py / "
                           "scripts/build_real_higgs_caches.py "
                           "(受控基准图与真实 SNAP Higgs 诱导子图缓存)")
                items.append(("data/benchmarks/" + fname, cmd))
    # 统一 LLM 后端的内容寻址归档：每条记录都是一次真实 API 调用的完整收据
    # （模型标识 / temperature / input-output-thinking token / response id / stop reason），
    # 同时充当缓存，因此也是「论文数字来自真实调用」的证据链。
    arch_dir = os.path.join(ROOT, "data", "llm_archive")
    if os.path.isdir(arch_dir):
        for model_dir in sorted(os.listdir(arch_dir)):
            md = os.path.join(arch_dir, model_dir)
            if not os.path.isdir(md):
                continue
            for fname in sorted(os.listdir(md)):
                if fname.endswith(".json"):
                    items.append((f"data/llm_archive/{model_dir}/{fname}",
                                  "src/network_evolve/llm_backend.py LLMClient.complete "
                                  "(内容寻址归档 = 一次真实 API 调用的收据，同时充当重跑缓存)"))
    return items


def build_manifest(previous=None):
    previous = previous or {}
    entries = {}
    for rel, cmd in collect_artifacts():
        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            continue
        st = os.stat(full)
        prev = previous.get(rel, {})
        entries[rel] = {
            "command": cmd or prev.get("command"),
            "sha256": sha256(full),
            "bytes": st.st_size,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": git("rev-parse", "HEAD"),
        }
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in ("numpy", "scipy", "networkx", "matplotlib"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[pkg] = "missing"
    dirty_lines = [line.strip() for line in git("status", "--porcelain").splitlines()
                   if line.strip() and not line.strip().endswith("results/PROVENANCE.json")]
    return {
        "schema": "network-evolve/provenance/1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(dirty_lines),
        "environment": env,
        "artifacts": entries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    previous = {}
    if os.path.exists(MANIFEST):
        try:
            previous = json.load(open(MANIFEST, encoding="utf-8")).get("artifacts", {})
        except Exception:
            previous = {}

    manifest = build_manifest(previous)
    artifacts = manifest["artifacts"]
    missing_cmd = [k for k, v in artifacts.items() if not v.get("command")]

    if a.check:
        problems = []
        for rel, rec in artifacts.items():
            full = os.path.join(ROOT, rel)
            if not os.path.exists(full):
                problems.append(f"{rel}: 文件缺失")
                continue
            cur = sha256(full)
            if cur != rec["sha256"]:
                problems.append(f"{rel}: SHA256 与清单不一致（结果被改动但未更新清单）")
        if missing_cmd:
            problems.append(f"以下产物缺少生成命令记录: {missing_cmd}")
        if problems:
            print("[FAIL] 溯源检查未通过：")
            for p in problems:
                print("  -", p)
            sys.exit(1)
        print(f"[PASS] {len(artifacts)} 个结果产物溯源一致")
        return

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[write] {os.path.relpath(MANIFEST, ROOT)}  ({len(artifacts)} 个产物)")
    if missing_cmd:
        print("[warn] 缺少生成命令记录:", missing_cmd)


if __name__ == "__main__":
    main()
