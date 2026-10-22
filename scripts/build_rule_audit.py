#!/usr/bin/env python3
"""Rule audit using a policy supplied by the outsourced annotator.

WARNING: this produces labels by mechanically applying a rule.  It is not
per-row human annotation and must not be reported as human validation or as
inter-annotator agreement.  It is a rule-based reference check.

Rule (provided by the outsourced annotator, post hoc):
  * both neutral -> keep
  * same direction -> keep with prob 0.8, mute with prob 0.2
  * one neutral:
      - other strongly for/against -> unfollow
      - other weak/moderate/neutral -> keep or mute (0.5/0.5)
  * both non-neutral and opposite:
      - both strong -> block with prob 0.9, unfollow with prob 0.1
      - otherwise -> mute with prob 0.6, unfollow with prob 0.4

"Reasonable" is defined as membership in the rule's allowed set (not necessarily
the sampled preferred action); confidence is sampled from {4,5}.
"""
import argparse
import csv
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HV = os.path.join(ROOT, "human_validation")

ACTION_ZH = {"keep": "保持", "mute": "静音", "unfollow": "取关", "block": "拉黑"}


def stance_of(text):
    m = re.search(r"\[(-?\d+(?:\.\d+)?)\]", text or "")
    return float(m.group(1)) if m else 0.0


def intensity(x):
    ax = abs(x)
    if ax <= 0.1:
        return "neutral"
    if ax <= 0.6:
        return "moderate"
    return "strong"


def sign(x):
    return 0 if abs(x) <= 0.1 else (1 if x > 0 else -1)


def peer_parse(message):
    t = (message or "").strip()
    for p in ("Just my two cents: ", "Trying to see both sides: ", "Announcing loudly: "):
        if t.startswith(p):
            t = t[len(p):]
            break
    if "neutral about" in t:
        return "neutral", 0
    m = re.search(r"I (weakly|moderately|strongly) strongly (for|against)", t)
    if not m:
        return "moderate", 0
    deg = {"weakly": "weak", "moderately": "moderate", "strongly": "strong"}[m.group(1)]
    return deg, (1 if m.group(2) == "for" else -1)


def classify(x, message):
    li = intensity(x)
    ld = sign(x)
    pi, pd = peer_parse(message)
    if li == "neutral" and pi == "neutral":
        return "both_neutral", {"keep"}
    if ld != 0 and pd != 0 and ld == pd:
        return "same_direction", {"keep", "mute"}
    if li == "neutral" or pi == "neutral":
        other = pi if li == "neutral" else li
        if other == "strong":
            return "neutral_vs_strong", {"unfollow"}
        return "neutral_vs_weak_moderate", {"keep", "mute"}
    if ld != pd:  # opposite and neither neutral
        if li == "strong" and pi == "strong":
            return "strong_opposite", {"block", "unfollow"}
        return "opposite_other", {"mute", "unfollow"}
    return "same_direction", {"keep", "mute"}


def sample_preferred(category, rng):
    if category == "both_neutral":
        return "keep"
    if category == "same_direction":
        return rng.choice(["keep"] * 8 + ["mute"] * 2)
    if category == "neutral_vs_strong":
        return "unfollow"
    if category == "neutral_vs_weak_moderate":
        return rng.choice(["keep", "mute"])
    if category == "strong_opposite":
        return "block" if rng.random() < 0.9 else "unfollow"
    if category == "opposite_other":
        return "mute" if rng.random() < 0.6 else "unfollow"
    return "keep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = list(csv.DictReader(open(os.path.join(HV, "annotation_sheet.csv"), encoding="utf-8")))
    key = {r["event_uid"]: r for r in csv.DictReader(open(os.path.join(HV, "annotation_key.csv"), encoding="utf-8"))}
    out = []
    counts = {}
    for r in rows:
        x = stance_of(r["listener_stance"])
        category, allowed = classify(x, r["peer_message"])
        preferred = sample_preferred(category, rng)
        model_action = key[r["event_uid"]]["model_action"]
        out.append({
            "event_uid": r["event_uid"],
            "source_model": key[r["event_uid"]]["source_model"],
            "listener_stance": x,
            "listener_intensity": intensity(x),
            "peer_category": peer_parse(r["peer_message"])[0],
            "rule_category": category,
            "rule_allowed": "|".join(sorted(allowed)),
            "rule_preferred_action": preferred,
            "model_action": model_action,
            "is_action_reasonable_synthetic": "合理" if model_action in allowed else "不合理",
            "confidence_1_5": rng.choice([4, 5]),
        })
        counts[category] = counts.get(category, 0) + 1
    path = os.path.join(HV, "rule_audit", "rule_audit_labels.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    # report
    def rate(rows, keyfn):
        n = len(rows); return (sum(1 for r in rows if keyfn(r)) / n if n else None)
    by_model = {}
    for model in sorted({r["source_model"] for r in out}):
        rs = [r for r in out if r["source_model"] == model]
        by_model[model] = {
            "n": len(rs),
            "reasonable_rate": rate(rs, lambda r: r["is_action_reasonable_synthetic"] == "合理"),
            "preferred_match_rate": rate(rs, lambda r: r["model_action"] == r["rule_preferred_action"]),
        }
    by_cat = {}
    for cat in sorted(counts):
        rs = [r for r in out if r["rule_category"] == cat]
        by_cat[cat] = {"n": len(rs),
                       "reasonable_rate": rate(rs, lambda r: r["is_action_reasonable_synthetic"] == "合理")}
    report = {"n": len(out), "seed": a.seed,
              "warning": "Rule supplied by outsourced annotator, mechanically applied; NOT per-row human annotation",
              "rule_counts": counts, "by_model": by_model, "by_rule_category": by_cat,
              "model_action_distribution": {m: {a2: sum(1 for r in out if r["source_model"] == m and r["model_action"] == a2)
                                                  for a2 in ("keep", "mute", "unfollow", "block")}
                                              for m in by_model}}
    with open(os.path.join(HV, "rule_audit", "rule_audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
