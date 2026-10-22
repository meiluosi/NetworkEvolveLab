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
    print(f"[write] {blind} ({len(picked)} rows)")
    print(f"[write] {key} ({len(picked)} rows)")


if __name__ == "__main__":
    main()
