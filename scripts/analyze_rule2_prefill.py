#!/usr/bin/env python3
"""Analyze the second rule-prefilled sheet (rule supplied by outsourced annotator).

This is a RULE AUDIT, not human annotation. It only summarizes the mechanically
prefilled labels and model actions for internal comparison.
"""
import csv
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HV = os.path.join(ROOT, "human_validation")
ACTIONS = {"保持": "keep", "静音": "mute", "取关": "unfollow", "拉黑": "block"}


def main():
    rows = list(csv.DictReader(open(os.path.join(HV, "annotation_sheet_zh_rule2_prefilled.csv"), encoding="utf-8-sig")))
    key = {r["event_uid"]: r for r in csv.DictReader(open(os.path.join(HV, "annotation_key_rule2.csv"), encoding="utf-8"))}
    by_model = defaultdict(list)
    by_cat = defaultdict(list)
    for r in rows:
        k = key[r["事件编号"]]
        rec = {"model": k["source_model"], "action": k["model_action"],
               "reasonable": r["动作是否合理"], "preferred": ACTIONS.get(r["你会选择的动作"], r["你会选择的动作"]),
               "cat": r["预填说明"].split("；")[0].replace("规则类别=", "")}
        by_model[rec["model"]].append(rec)
        by_cat[rec["cat"]].append(rec)

    def summarize(rs):
        n = len(rs)
        return {
            "n": n,
            "reasonable_rate": sum(1 for r in rs if r["reasonable"] == "合理") / n if n else None,
            "preferred_match_rate": sum(1 for r in rs if r["action"] == r["preferred"]) / n if n else None,
            "model_actions": dict(Counter(r["action"] for r in rs)),
            "prefilled_actions": dict(Counter(r["preferred"] for r in rs)),
        }
    report = {
        "warning": "Rule supplied by outsourced annotator, mechanically applied; NOT per-row human annotation",
        "n": len(rows),
        "by_model": {m: summarize(rs) for m, rs in by_model.items()},
        "by_rule_category": {c: summarize(rs) for c, rs in by_cat.items()},
    }
    out = os.path.join(HV, "rule_audit", "rule2_audit_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    lines = ["# 第二套外包规则审计（合成，非逐条人工标注）", "",
             "**警告：这些标签按外包方规则机械预填，不是人工逐条判断，不能写成 human validation。**", "",
             f"- 样本数：{report['n']}", ""]
    for m, v in report["by_model"].items():
        lines.append(f"- {m}: n={v['n']}, 规则合理率={v['reasonable_rate']:.3f}, 与预填偏好动作一致率={v['preferred_match_rate']:.3f}")
    lines += ["", "## 按规则类别", ""]
    for c, v in sorted(report["by_rule_category"].items()):
        lines.append(f"- {c}: n={v['n']}, 规则合理率={v['reasonable_rate']:.3f}")
    with open(os.path.join(HV, "rule_audit", "rule2_audit_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
