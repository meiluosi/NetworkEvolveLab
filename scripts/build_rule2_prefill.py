#!/usr/bin/env python3
"""Build a disjoint 300-event sample and a rule-2-prefilled review sheet.

The rule is supplied by the outsourced annotator. This output is a PREFILLED
DRAFT for human review, not human annotation.

Rule-2 interpretation used here (documented in the package):
  a = listener, b = peer; intensity: neutral |x|<=0.1, moderate <=0.6, strong.
  a neutral, b opposite:  b strong -> unfollow 90% / mute 10%; b weak/moderate -> mute.
  a moderate, b opposite: b strong -> block; b moderate -> unfollow; b weak/neutral -> mute.
  a strong, b opposite:   does not block immediately; argues (keep/mute), then blocks.
                          preferred sampled {keep 50%, mute 30%, block 20%}; allowed {keep, mute, block}.
  same camp:              a strong -> keep; otherwise -> mute.
  both neutral -> keep; one neutral otherwise -> mute.
"""
import argparse
import csv
import glob
import importlib.util
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from build_human_validation_sample import (  # noqa: E402
    ARCH_ZH, ACTION_ZH, MODELS, zh_message, zh_profile,
)
from build_rule_audit import intensity, peer_parse, sign, stance_of  # noqa: E402


def parse(path):
    spec = importlib.util.spec_from_file_location("hv_builder", os.path.join(ROOT, "scripts", "build_human_validation_sample.py"))
    # use the imported function directly; kept simple below
    return None


def rule2(listener, message):
    li, ld = intensity(listener), sign(listener)
    pi, pd = peer_parse(message)
    same = (ld != 0 and pd != 0 and ld == pd)
    opp = (ld != 0 and pd != 0 and ld != pd)
    if li == "neutral" and pi == "neutral":
        return "both_neutral", {"keep"}, "keep"
    if same:
        if li == "strong":
            return "same_strong", {"keep"}, "keep"
        return "same_other", {"mute"}, "mute"
    if li == "neutral":
        if opp and pi == "strong":
            return "neutral_vs_strong_opposite", {"unfollow", "mute"}, "unfollow"
        return "neutral_vs_other", {"mute"}, "mute"
    if li == "moderate":
        if opp and pi == "strong":
            return "moderate_vs_strong_opposite", {"block"}, "block"
        if opp and pi == "moderate":
            return "moderate_vs_moderate_opposite", {"unfollow"}, "unfollow"
        return "moderate_vs_weak_or_neutral", {"mute"}, "mute"
    if li == "strong":
        if opp:
            return "strong_vs_opposite", {"keep", "mute", "block"}, "argue_then_block"
        return "strong_vs_other", {"mute"}, "mute"
    return "fallback", {"mute"}, "mute"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # Exclusions from the first sample.
    old_rows = list(csv.DictReader(open(os.path.join(ROOT, "human_validation", "annotation_sheet.csv"), encoding="utf-8")))
    old_key = {r["event_uid"]: r for r in csv.DictReader(open(os.path.join(ROOT, "human_validation", "annotation_key.csv"), encoding="utf-8"))}
    old = {(old_key[r["event_uid"]]["source_receipt"], r["listener_profile"], r["peer_message"], old_key[r["event_uid"]]["model_action"])
           for r in old_rows if r["event_uid"] in old_key}
    picked = []
    for model, meta in MODELS.items():
        rows = []
        for path in glob.glob(os.path.join(ROOT, "data", "llm_archive", model, "*.json")):
            import json
            d = json.load(open(path, encoding="utf-8"))
            if int(d.get("n_events", 0)) != meta["n_events"] or int(d.get("max_tokens", 0)) != meta["max_tokens"]:
                continue
            try:
                prompt = d["prompt"]; js = json.loads(prompt[prompt.index("{"):]); dec = json.loads(d["response_text"])
            except Exception:
                continue
            ev = {e.get("id"): e for e in js.get("events", [])}
            for x in dec.get("decisions", []):
                e = ev.get(x.get("id"))
                if not e:
                    continue
                k = (os.path.relpath(path, ROOT), e.get("your_profile"), e.get("peer_says"), x.get("action"))
                if k in old:
                    continue
                rows.append({"event_uid": None, "source_model": model,
                             "listener_archetype": e.get("your_archetype"),
                             "listener_stance": e.get("your_stance"),
                             "listener_profile": e.get("your_profile"),
                             "peer_message": e.get("peer_says"),
                             "tie_strength": e.get("tie_strength"),
                             "listener_tolerance": e.get("your_tolerance"),
                             "model_action": x.get("action"),
                             "model_new_belief": json.dumps(x.get("new_belief")),
                             "model_reasoning": x.get("reasoning", ""),
                             "source_receipt": k[0]})
        nonkeep = [r for r in rows if r["model_action"] != "keep"]
        keep = [r for r in rows if r["model_action"] == "keep"]
        picked.extend(rng.sample(nonkeep, min(75, len(nonkeep))))
        picked.extend(rng.sample(keep, min(75, len(keep))))
    rng.shuffle(picked)
    for i, r in enumerate(picked, 1):
        r["event_uid"] = f"HR{i:04d}"
    hv = os.path.join(ROOT, "human_validation")
    # Chinese sheet
    cols = ["事件编号", "听众类型", "听众立场", "听众立场描述", "对方发言", "关系强度",
            "模型动作", "动作是否合理", "你会选择的动作", "把握程度1到5", "备注", "预填说明"]
    zh_action = {"keep": "保持", "mute": "静音", "unfollow": "取关", "block": "拉黑"}
    out = []
    for r in picked:
        x = stance_of(str(r["listener_stance"]))
        cat, allowed, sampled = rule2(x, r["peer_message"])
        if cat == "strong_vs_opposite":
            sampled = rng.choices(["keep", "mute", "block"], weights=[5, 3, 2])[0]
        reasonable = "合理" if r["model_action"] in allowed else "不合理"
        out.append({
            "事件编号": r["event_uid"],
            "听众类型": ARCH_ZH.get(r["listener_archetype"], r["listener_archetype"]),
            "听众立场": r["listener_stance"],
            "听众立场描述": zh_profile(r["listener_profile"]),
            "对方发言": zh_message(r["peer_message"]),
            "关系强度": r["tie_strength"],
            "模型动作": zh_action.get(r["model_action"], r["model_action"]),
            "动作是否合理": reasonable,
            "你会选择的动作": zh_action.get(sampled, sampled),
            "把握程度1到5": rng.choice([4, 5]),
            "备注": "",
            "预填说明": f"规则类别={cat}；允许集合={','.join(sorted(allowed))}；预填仅供参考，请逐条复核",
        })
    p = os.path.join(hv, "annotation_sheet_zh_rule2_prefilled.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
    # key
    kp = os.path.join(hv, "annotation_key_rule2.csv")
    with open(kp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event_uid", "source_model", "model_action", "model_new_belief", "model_reasoning", "source_receipt"])
        w.writeheader()
        for r in picked:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"[write] {p} ({len(out)} rows)")
    print(f"[write] {kp} ({len(picked)} rows)")


if __name__ == "__main__":
    main()
