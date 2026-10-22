# 第二套外包规则审计（合成，非逐条人工标注）

**警告：这些标签按外包方规则机械预填，不是人工逐条判断，不能写成 human validation。**

- 样本数：300

- deepseek-flash: n=150, 规则合理率=0.573, 与预填偏好动作一致率=0.420
- MiniMax-M3: n=150, 规则合理率=0.640, 与预填偏好动作一致率=0.440

## 按规则类别

- moderate_vs_moderate_opposite: n=26, 规则合理率=0.192
- moderate_vs_strong_opposite: n=36, 规则合理率=0.000
- moderate_vs_weak_or_neutral: n=4, 规则合理率=0.250
- same_other: n=49, 规则合理率=0.061
- same_strong: n=99, 规则合理率=1.000
- strong_vs_opposite: n=86, 规则合理率=0.860
