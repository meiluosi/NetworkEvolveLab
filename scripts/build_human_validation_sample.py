#!/usr/bin/env python3
"""Build a blind 300-event human-validation sheet from high-coverage receipts.

Design
------
- 150 events from MiniMax-M3 and 150 from DeepSeek-flash (both 60-event arms).
- Per model: 75 non-keep decisions (all action types) + 75 random keep decisions.
- The annotator sheet hides model identity and model reasoning; the key file keeps
  source model, action, new belief, reasoning, and receipt path for analysis.
- Event order is shuffled; IDs are re-assigned (HV0001...).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCH = os.path.join(ROOT, "data", "llm_archive")
OUT = os.path.join(ROOT, "human_validation")
MODELS = {
    "MiniMax-M3": {"n_events": 60, "max_tokens": 8192},
    "deepseek-flash": {"n_events": 60, "max_tokens": 16384},
}


def parse_receipt(path):
    d = json.load(open(path, encoding="utf-8"))
    meta = MODELS.get(d.get("model"))
    if not meta:
        return []
    if int(d.get("n_events", 0)) != meta["n_events"] or int(d.get("max_tokens", 0)) != meta["max_tokens"]:
        return []
    try:
        prompt = d["prompt"]
        js = json.loads(prompt[prompt.index("{"):])
        dec = json.loads(d["response_text"])
    except (KeyError, ValueError):
        return []
    events = {e.get("id"): e for e in js.get("events", [])}
    out = []
    for x in dec.get("decisions", []):
        e = events.get(x.get("id"))
        if not e:
            continue
        out.append({
            "event_uid": None,
            "source_model": d.get("model"),
            "listener_archetype": e.get("your_archetype"),
            "listener_stance": json.dumps(e.get("your_stance")),
            "listener_profile": e.get("your_profile"),
            "peer_message": e.get("peer_says"),
            "tie_strength": e.get("tie_strength"),
            "listener_tolerance": e.get("your_tolerance"),
            "model_action": x.get("action"),
            "model_new_belief": json.dumps(x.get("new_belief")),
            "model_reasoning": x.get("reasoning", ""),
            "source_receipt": os.path.relpath(path, ROOT),
        })
    return out


ARCH_ZH = {"hub": "枢纽型（hub）", "bridger": "桥梁型（bridger）", "tail": "长尾型（tail）"}
ACTION_ZH = {"keep": "保持", "mute": "静音", "unfollow": "取关", "block": "拉黑"}
PREFIX_ZH = {"Just my two cents: ": "一点个人看法：",
             "Trying to see both sides: ": "尽量兼顾双方：",
             "Announcing loudly: ": "大声宣布："}
DEGREE_ZH = {"weakly": "轻度", "moderately": "中等程度", "strongly": "强烈"}


def zh_profile(text: str) -> str:
    t = text.strip()
    if t.startswith("is neutral about"):
        return "对争议政策轴保持中立"
    m = re.match(r"(weakly|moderately|strongly) strongly (for|against) the incumbent policy directions the contested policy axis", t)
    if m:
        return f"{DEGREE_ZH[m.group(1)]}{'支持' if m.group(2) == 'for' else '反对'}现任政策方向（争议政策轴）"
    return t


def zh_message(text: str) -> str:
    t = text.strip()
    prefix = ""
    for p, z in PREFIX_ZH.items():
        if t.startswith(p):
            prefix, t = z, t[len(p):]
            break
    if t.startswith("I am neutral about the contested policy axis."):
        return prefix + "我对争议政策轴保持中立。"
    m = re.match(r"I (weakly|moderately|strongly) strongly (for|against) the incumbent policy direction the contested policy axis\.", t)
    if m:
        return prefix + f"我{DEGREE_ZH[m.group(1)]}{'支持' if m.group(2) == 'for' else '反对'}现任政策方向（争议政策轴）。"
    return prefix + t


def write_zh(picked, out_dir):
    cols = ["事件编号", "听众类型", "听众立场", "听众立场描述", "对方发言",
            "关系强度", "模型动作", "动作是否合理", "你会选择的动作",
            "把握程度1到5", "备注"]
    path = os.path.join(out_dir, "annotation_sheet_zh.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in picked:
            w.writerow({
                "事件编号": r["event_uid"],
                "听众类型": ARCH_ZH.get(r["listener_archetype"], r["listener_archetype"]),
                "听众立场": r["listener_stance"],
                "听众立场描述": zh_profile(r["listener_profile"]),
                "对方发言": zh_message(r["peer_message"]),
                "关系强度": r["tie_strength"],
                "模型动作": ACTION_ZH.get(r["model_action"], r["model_action"]),
                "动作是否合理": "",
                "你会选择的动作": "",
                "把握程度1到5": "",
                "备注": "",
            })
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-model-nonkeep", type=int, default=75)
    ap.add_argument("--per-model-keep", type=int, default=75)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    picked = []
    for model in MODELS:
        rows = []
        for path in glob.glob(os.path.join(ARCH, model, "*.json")):
            rows.extend(parse_receipt(path))
        nonkeep = [r for r in rows if r["model_action"] != "keep"]
        keep = [r for r in rows if r["model_action"] == "keep"]
        # Deterministic sample; dedupe on (receipt, profile, message, action).
        def uniq(xs):
            seen, out = set(), []
            for r in xs:
                k = (r["source_receipt"], r["listener_profile"], r["peer_message"], r["model_action"])
                if k not in seen:
                    seen.add(k); out.append(r)
            return out
        nonkeep = uniq(nonkeep); keep = uniq(keep)
        picked.extend(rng.sample(nonkeep, min(a.per_model_nonkeep, len(nonkeep))))
        picked.extend(rng.sample(keep, min(a.per_model_keep, len(keep))))
    rng.shuffle(picked)
    for i, r in enumerate(picked, 1):
        r["event_uid"] = f"HV{i:04d}"
    os.makedirs(OUT, exist_ok=True)
    blind = os.path.join(OUT, "annotation_sheet.csv")
    key = os.path.join(OUT, "annotation_key.csv")
    blind_cols = ["event_uid", "listener_archetype", "listener_stance", "listener_profile",
                  "peer_message", "tie_strength", "model_action",
                  "is_action_reasonable", "preferred_action", "confidence_1_5", "comments"]
    with open(blind, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=blind_cols, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            row = {k: r[k] for k in blind_cols if k in r}
            for k in ("is_action_reasonable", "preferred_action", "confidence_1_5", "comments"):
                row[k] = ""
            w.writerow(row)
    key_cols = ["event_uid", "source_model", "model_action", "model_new_belief",
                "model_reasoning", "source_receipt"]
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_cols, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            w.writerow(r)
    zh = write_zh(picked, OUT)
    print(f"[write] {blind} ({len(picked)} rows)")
    print(f"[write] {key} ({len(picked)} rows)")
    print(f"[write] {zh} ({len(picked)} rows)")


if __name__ == "__main__":
    main()
