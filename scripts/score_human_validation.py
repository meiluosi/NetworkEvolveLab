#!/usr/bin/env python3
"""Score two completed human-validation sheets against the model key.

No annotations are bundled with the repository; this script only runs after the
annotators return the CSVs. It reports inter-annotator agreement, human-vs-model
action agreement, and reasonableness by model/action. It never silently drops
rows: unusable rows are counted and listed.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

REASONABLE = {"reasonable", "unreasonable", "cannot_judge"}
ACTIONS = {"keep", "mute", "unfollow", "block"}

# Chinese outsourcing sheet (annotation_sheet_zh.csv) support.
COL_ALIASES = {
    "事件编号": "event_uid",
    "动作是否合理": "is_action_reasonable",
    "你会选择的动作": "preferred_action",
    "把握程度1到5": "confidence_1_5",
    "备注": "comments",
}
VALUE_ALIASES = {
    "合理": "reasonable", "不合理": "unreasonable", "无法判断": "cannot_judge",
    "保持": "keep", "静音": "mute", "取关": "unfollow", "拉黑": "block",
}


def normalize_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[COL_ALIASES.get(k, k)] = VALUE_ALIASES.get((v or "").strip(), v)
    return out


def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        # All observed labels are one class: kappa is undefined, not zero.
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def read_sheet(path: str) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [normalize_row(r) for r in csv.DictReader(f)]
    return {r.get("event_uid", "").strip(): r for r in rows if r.get("event_uid")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", default="human_validation/human_validation_report")
    a = ap.parse_args()
    A, B = read_sheet(a.a), read_sheet(a.b)
    key = read_sheet(a.key)
    ids = sorted(set(A) & set(B) & set(key))
    missing = {"in_A_not_B": sorted(set(A) - set(B)),
               "in_B_not_A": sorted(set(B) - set(A)),
               "missing_key": sorted((set(A) | set(B)) - set(key))}
    reasonable_A, reasonable_B, action_A, action_B = [], [], [], []
    unusable = []
    disagreements = []
    by_model = defaultdict(lambda: Counter())
    by_model_action = defaultdict(lambda: Counter())
    for uid in ids:
        ra = (A[uid].get("is_action_reasonable") or "").strip()
        rb = (B[uid].get("is_action_reasonable") or "").strip()
        pa = (A[uid].get("preferred_action") or "").strip()
        pb = (B[uid].get("preferred_action") or "").strip()
        if ra not in REASONABLE or rb not in REASONABLE or pa not in ACTIONS or pb not in ACTIONS:
            unusable.append(uid)
            continue
        reasonable_A.append(ra); reasonable_B.append(rb)
        action_A.append(pa); action_B.append(pb)
        model = key[uid].get("source_model", "?")
        mact = key[uid].get("model_action", "?")
        for label in (ra, rb):
            by_model[model][label] += 1
            by_model_action[(model, mact)][label] += 1
        if ra != rb or pa != pb:
            disagreements.append({"event_uid": uid, "A": [ra, pa], "B": [rb, pb]})
    # human-vs-model: use A and B separately; also consensus rows.
    model_actions = [key[u].get("model_action", "") for u in ids if u not in unusable]
    human_actions = action_A  # aligned with usable ids
    def agreement(pred):
        pairs = [(p, m) for p, m in zip(pred, model_actions) if p and m]
        return sum(1 for p, m in pairs if p == m) / len(pairs) if pairs else None
    conf_A = Counter((p, m) for p, m in zip(action_A, model_actions))
    report = {
        "n_rows_compared": len(reasonable_A),
        "n_unusable": len(unusable),
        "unusable_event_uids": unusable[:200],
        "missing": {k: v[:200] for k, v in missing.items()},
        "inter_annotator": {
            "reasonable": cohen_kappa(reasonable_A, reasonable_B),
            "preferred_action": cohen_kappa(action_A, action_B),
            "raw_agreement_reasonable": (sum(1 for x, y in zip(reasonable_A, reasonable_B) if x == y) / len(reasonable_A) if reasonable_A else None),
            "raw_agreement_action": (sum(1 for x, y in zip(action_A, action_B) if x == y) / len(action_A) if action_A else None),
        },
        "human_vs_model": {
            "action_agreement_A": agreement(action_A),
            "action_agreement_B": agreement(action_B),
            "action_agreement_consensus": agreement([x if x == y else "" for x, y in zip(action_A, action_B)]),
        },
        "by_source_model": {m: dict(c) for m, c in by_model.items()},
        "by_model_action": {f"{m}/{act}": dict(c) for (m, act), c in by_model_action.items()},
        "n_disagreement_rows": len(disagreements),
        "disagreements_sample": disagreements[:50],
        "confusion_preferred_vs_model_A": {f"{p}->{m}": c for (p, m), c in sorted(conf_A.items())},
    }
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    lines = ["# Human validation report", "",
             f"- compared rows: {report['n_rows_compared']}",
             f"- unusable rows: {report['n_unusable']}",
             f"- missing: {report['missing']}",
             f"- inter-annotator Cohen's kappa (reasonable): {report['inter_annotator']['reasonable']}",
             f"- inter-annotator Cohen's kappa (preferred action): {report['inter_annotator']['preferred_action']}",
             f"- human-vs-model action agreement A: {report['human_vs_model']['action_agreement_A']}",
             f"- human-vs-model action agreement B: {report['human_vs_model']['action_agreement_B']}",
             f"- rows with any disagreement: {report['n_disagreement_rows']}", ""]
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[write] {a.out}.json")
    print(f"[write] {a.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
