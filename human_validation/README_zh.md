# 人工校验外包包（中文版，300 条）

## 发给外包的文件

**只发这个压缩包：**

`human_validation/outsourcing_package_zh.zip`

里面只有：

- `annotation_sheet_zh.csv`：300 条中文标注表；
- `codebook_zh.md`：中文判定规则；
- `ANNOTATOR_README_zh.md`：中文操作说明。

## 不要发

- `annotation_key.csv`：含模型身份、模型 reasoning、来源收据，只给研究团队分析用；
- 论文原文、假设、MiniMax/DeepSeek 结果。

## 样本设计

- 300 条：MiniMax-M3 150 条 + DeepSeek-flash 150 条；
- 每个模型 75 条非“保持”动作 + 75 条随机“保持”；
- 顺序已打乱，事件编号匿名化为 HV0001–HV0300；
- 标注者只看到听众立场、类型、对方发言、关系强度、模型动作；
- 看不到模型名、模型 reasoning、实验分组。

## 回收后怎么评分

把两份填好的表放回 `human_validation/`，例如：

```bash
python scripts/score_human_validation.py \
  --a human_validation/annotation_sheet_zh_标注者A.csv \
  --b human_validation/annotation_sheet_zh_标注者B.csv \
  --key human_validation/annotation_key.csv \
  --out human_validation/human_validation_report
```

评分脚本会自动识别中文列名和中文答案（合理/不合理/无法判断、保持/静音/取关/拉黑），不需要你手动改回英文。

## 建议的验收标准

- 两人都完成 300 条，且前两列无空值；
- 两人“是否合理”的 Cohen's κ ≥ 0.6 才作为强人工验证；
- κ < 0.6 只能写成 inconclusive，不能硬说人工验证通过；
- 保留分歧样本，做简短仲裁记录。


## 规则审计（不要把规则审计当成人类标注）

`human_validation/rule_audit/` 是按**外包方提供的规则**机械生成的参照标签，只用于内部诊断。它不是外包逐条人工标注，不能写成 human validation，也不能算人工 κ。如果外包实际逐条标注了 300 条，请以回收的标注表为准。


## 预填审校包（推荐给外包）

如果外包已经给出规则，可以先发预填审校包：

`human_validation/outsourcing_review_package_zh.zip`

里面是规则预填好的 `annotation_sheet_zh_prefilled.csv`、codebook 和审校说明。外包逐条复核/修改后交回；论文里应写成 **human-reviewed rule-prefilled audit**，并披露锚定效应，不能写成完全独立的人工标注。


## 第二套规则

第二套外包规则已生成新的 300 条（与第一套不重复）：

`human_validation/outsourcing_review_package_rule2_zh.zip`

包含 `annotation_sheet_zh_rule2_prefilled.csv`、通用 codebook、`RULE2_REVIEWER_README_zh.md`。外包逐条复核后返回两个 CSV。


## 第二套规则审计

第二套规则预填的内部审计结果：

- [rule2_audit_report.md](rule_audit/rule2_audit_report.md)
- [rule2_audit_report.json](rule_audit/rule2_audit_report.json)

这是按外包方规则机械预填的 rule audit，不是人工逐条标注；不能写成 human validation。
