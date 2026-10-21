"""Phase 1b：Bridger 受压机制消融(H2) + block_diff 扫描 + T>=50 的 H1 判定。固定种子 0..N-1，95% CI。
用法: python -m src.network_evolve.phase1b --seeds 20 --out results/phase1b.json
"""
import argparse
import json
from .expkit import run_many, fmt, h2_summary, h1_summary

log_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--out", default="results/phase1b.json")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    R = a.rounds
    out = {"args": vars(a), "h2_ablation": {}, "block_sweep": {}, "h1_grid": {}}

    P(f"## (A) H2 消融：Bridger 机制（theta=0.4, block=1.2, T={R}, seeds={a.seeds}）")
    P("config | bridger_ret | other_ret | gap(rel) [CI95] | p | H2")
    confs = {
        "base(原模型,bridger_tol=0.7)": dict(),
        "base,bridger_tol=θ(无天然包容优势)": dict(bridger_dtheta=0.0),
        "moderate(结构)": dict(bridger_moderate=True),
        "stress(动力学)": dict(bridger_stress=True),
        "moderate+stress sens=0.3": dict(bridger_moderate=True, bridger_stress=True, stress_sens=0.3),
        "moderate+stress sens=0.6": dict(bridger_moderate=True, bridger_stress=True, stress_sens=0.6),
        "moderate+stress sens=0.9": dict(bridger_moderate=True, bridger_stress=True, stress_sens=0.9),
        "moderate+stress sens=0.9,bridger_tol=θ": dict(bridger_moderate=True, bridger_stress=True,
                                                       stress_sens=0.9, bridger_dtheta=0.0),
    }
    for name, c in confs.items():
        runs = run_many({**c, "block": 1.2, "rounds": R}, seeds)
        h = h2_summary(runs)
        out["h2_ablation"][name] = h
        P(f"{name} | {h['bridger_mean']:.3f} | {h['other_mean']:.3f} | {h['relative_gap']:+.3f} "
          f"[{h['gap_ci95'][0]:+.3f},{h['gap_ci95'][1]:+.3f}] | {h['p_value']:.3g} | {h['H2_supported']}")

    P(f"\n## (B) block_diff 扫描（T={R}）：Louvain Q_end / giant / blocks / cross")
    P("model | block | Q_end | giant | blocks | cross | H2gap")
    for model, c in {"base": dict(), "moderate+stress": dict(bridger_moderate=True, bridger_stress=True,
                                                              stress_sens=0.9)}.items():
        for blk in (None, 0.8, 1.0, 1.2, 1.4, 1.6):
            runs = run_many({**c, "block": blk, "rounds": R}, seeds)
            g = h2_summary(runs)["relative_gap"]
            row = dict(q_end=fmt([r["q_end"] for r in runs]), giant=fmt([r["giant"] for r in runs]),
                       blocks=fmt([r["blocks"] for r in runs]), cross=fmt([r["cross"] for r in runs]), h2gap=g)
            out["block_sweep"][f"{model}|{blk}"] = row
            P(f"{model} | {blk} | {row['q_end']} | {row['giant']} | {row['blocks']} | {row['cross']} | {g:+.3f}")

    P("\n## (C) H1 判定网格（T=100；Q_init<0.25 且 Q_end>0.65 且 >=2 同质回音室）")
    P("theta | block | p_form | Q_init | Q_end | max Q_end | homo_comms | H1通过率")
    for th in (0.2, 0.3, 0.4):
        for blk in (None, 1.2):
            for pf in (0.1, 0.3):
                runs = run_many(dict(theta=th, block=blk, p_form=pf, rounds=100,
                                     checkpoints=(10, 25, 50, 75, 100)), seeds)
                h = h1_summary(runs)
                out["h1_grid"][f"{th}|{blk}|{pf}"] = h
                P(f"{th} | {blk} | {pf} | {h['q_init'][0]:.3f} | {h['q_end'][0]:.3f}±{h['q_end'][1]:.3f} | "
                  f"{h['max_q_end']:.3f} | {h['homo_comms'][0]:.2f} | {h['h1_pass_rate']:.2f}")
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    open(a.out.replace(".json", ".log"), "w").write("\n".join(log_lines))


if __name__ == "__main__":
    main()
