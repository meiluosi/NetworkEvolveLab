# Outsourced-annotator rule audit（合成，非逐条人工标注）

**警告：这些标签由外包方提供的规则机械生成，不是外包逐条人工标注；不能写成 human validation，也不能算人工 κ。若外包实际逐条标注了 300 条，请以回收的标注表为准。**

- 样本数：300
- 规则类别计数：{'opposite_other': 122, 'same_direction': 128, 'strong_opposite': 37, 'neutral_vs_strong': 2, 'neutral_vs_weak_moderate': 11}

- MiniMax-M3: n=150, 规则合理率=0.847, 与采样偏好动作一致率=0.540
- deepseek-flash: n=150, 规则合理率=0.840, 与采样偏好动作一致率=0.520

## 按规则类别

- neutral_vs_strong: n=2, 规则合理率=0.000
- neutral_vs_weak_moderate: n=11, 规则合理率=0.818
- opposite_other: n=122, 规则合理率=0.861
- same_direction: n=128, 规则合理率=1.000
- strong_opposite: n=37, 规则合理率=0.297

## 关键发现

两个模型在 `strong_opposite`（双方强烈对立）类别中只有 29.7% 的动作落在规则允许集合（block/unfollow）内；在同向类别中为 100%。这说明模型在强对立场景下比外包方规则更少采取断边动作，但这只是 rule audit，不是人类判断。
