# 四阶段实验推进计划 (Experiment Plan)

延续 WorldSimulationLab 的严格科研准则：**必须保留失败记录、实施消融对照、固定哈希随机种子、严格控制 API 成本**。

---

## 阶段 0：离线统计动力学基线核验（Phase 0: Offline Synthetic Baseline）
- **目标**：在不消耗任何 API Key 算力的情况下，用连续动力学数值方程（DeGroot, Hegselmann-Krause, Bounded Confidence）跑通 $N=100$ 节点的动态拓扑演化闭环。
- **产出**：
  - 确认模块度 $Q$、极化指数 $\Pi$、网络密度随时间步变化的演化曲线。
  - 确定拓扑相变的理论边界条件（临界阈值 $\theta^*$）。

---

## 阶段 1：异质原型接入与语义失调实验（Phase 1: Archetypes & Semantic Divergence）
- **目标**：接入 3 类核心行为原型（Top 2% Hub, 8.5% Bridger, 89.5% Long-tail）。
- **实验设计**：
  - **对照组 A（同质基线）**：全部智能体使用统一 Prompt，单纯重复抽样。
  - **实验组 B（异质原型）**：Hub 频繁设置议程，Bridger 承担跨域分发，Long-tail 做出反馈。
  - **检验指标**：异质原型是否加速了社群分化？Bridger 节点的断连速率是否最高？

---

## 阶段 2：大模型语义真机核验与轨迹冻结（Phase 2: LLM Real Verification & Freeze）
- **预算控制**：总调用次数严格控制在 **100 次以内的真实 LLM（如 DeepSeek）调用**。
- **任务**：针对具有代表性的争议话题事件流（Controversial Topic Stream），抽取 20 轮真实演化。
- **产出**：
  - 冻结的 JSON 轨迹文件与 SQLite 数据库；
  - 导出动态网络结构 SVG 演化动画与交互式热力图。

---

## 阶段 3：反事实去极化政策干预（Phase 3: Counterfactual Policy Interventions）
- **目标**：评估 3 种平台治理政策对抑制极化和回音室的有效性：
  1. **Policy 1（去中心化推荐，Diversified RecSys）**：强制向用户推荐 15% 异质但高质量的对立观点。
  2. **Policy 2（桥接者权重扶持，Bridger Subsidy）**：对 `active_bridger` 的跨群中立言论降低互动摩擦力。
  3. **Policy 3（热点降温阻尼，Hub Rate Limiting）**：对极端情绪化 Hub 节点实施 50% 的信息传播限流。
- **评估输出**：各策略下的极化缓解率、信息通达度损失、以及级联阻隔乘数。
