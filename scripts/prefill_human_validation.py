#!/usr/bin/env python3
"""Fill the Chinese annotation sheet with the outsourced-annotator rule draft.

The output is a PREFILLED DRAFT, not human annotation. Human reviewers must
review every row; the paper must describe the result as human-reviewed
rule-prefilled audit, not as independent annotation.
"""
import argparse
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HV = os.path.join(ROOT, "human_validation")
ACTION_ZH = {"keep": "保持", "mute": "静音", "unfollow": "取关", "block": "拉黑"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blank", default=os.path.join(HV, "annotation_sheet_zh.csv"))
    ap.add_argument("--rule", default=os.path.join(HV, "rule_audit", "rule_audit_labels.csv"))
    ap.add_argument("--out", default=os.path.join(HV, "annotation_sheet_zh_prefilled.csv"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.blank, encoding="utf-8-sig")))
    rule = {r["event_uid"]: r for r in csv.DictReader(open(a.rule, encoding="utf-8-sig"))}
    cols = ["事件编号", "听众类型", "听众立场", "听众立场描述", "对方发言",
            "关系强度", "模型动作", "动作是否合理", "你会选择的动作",
            "把握程度1到5", "备注", "预填说明"]
    out = []
    for r in rows:
        q = rule.get(r["事件编号"])
        if q is None:
            raise SystemExit(f"missing rule for {r['事件编号']}")
        d = dict(r)
        d["动作是否合理"] = "合理" if q["is_action_reasonable_synthetic"] == "合理" else "不合理"
        d["你会选择的动作"] = ACTION_ZH[q["rule_preferred_action"]]
        d["把握程度1到5"] = q["confidence_1_5"]
        d["备注"] = ""
        d["预填说明"] = (f"规则类别={q['rule_category']}；允许集合="
                        f"{q['rule_allowed'].replace('|','/')}；预填仅为草稿，请逐条复核")
        out.append({k: d.get(k, "") for k in cols})
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out)
    print(f"[write] {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
