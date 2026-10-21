# 架构与动力学数学模型设计 (Architecture & Mathematical Framework)

## 1. 核心数学形式化

定义动态时变网络 $G(t) = (V, E(t), W(t))$：
- 节点集合 $V = \{v_1, v_2, \dots, v_N\}$，每个节点对应一个大模型智能体。
- 智能体状态元组：$S_i(t) = \langle \mathbf{b}_i(t), \tau_i, C_i(t) \rangle$
  - $\mathbf{b}_i(t) \in [-1, +1]^d$：连续语义信念/观点嵌入向量（Belief State）。
  - $\tau_i \in \{\text{hub\_broadcaster}, \text{active\_bridger}, \text{long\_tail\_responder}\}$：行为原型类型。
  - $C_i(t)$：智能体记忆与上下文窗口。
- 动态拓扑边：$e_{ij}(t) \in E(t)$，权重 $w_{ij}(t) \in [0, 1]$ 表示节点 $i$ 对节点 $j$ 的关注强弱/信任权重。

### 1.1 认知与信念更新（微观动力学）
当节点 $i$ 在时间步 $t$ 接收到邻居节点 $j \in \mathcal{N}_i(t)$ 的发帖内容 $m_j(t)$ 时：
1. **语义认知共鸣度计算**：
   $$\text{Sim}(i, j) = \cos(\mathbf{b}_i(t), \mathbf{b}_j(t))$$
2. **阻尼信念更新（Damped Belief Shift）**：
   $$\mathbf{b}_i(t+1) = \mathbf{b}_i(t) + \eta \cdot w_{ij}(t) \cdot f_{\text{LLM}}(m_j(t), C_i(t))$$
   其中 $\eta \in (0, 0.2]$ 为阻尼学习率，严格防止大模型单次幻觉导致观点剧烈震荡。

### 1.2 关系拓扑动态演化（宏观拓扑演变）
在每个交互周期后，节点 $i$ 评估其与邻居 $j$ 的关系边权重 $w_{ij}$：
1. **认知失调衰减（Unfollow / Mute 机制）**：
   若 $\text{Sim}(i, j) < \theta_{\text{tolerance}}$，边权重衰减：
   $$w_{ij}(t+1) = \max(0, w_{ij}(t) - \gamma_{\text{decay}})$$
   当 $w_{ij} = 0$ 时，触发物理断连（Unfollow 动作）。
2. **同质性连边形成（Triadic Closure & Follow 机制）**：
   通过两跳邻居或推荐系统发现新节点 $k$，若 $\text{Sim}(i, k) > \theta_{\text{attract}}$，以概率 $P_{\text{form}}$ 建立新边 $e_{ik}$。

---

## 2. 宏观涌现观测指标（Macro Metrics）

1. **纽曼模块度（Newman's Modularity $Q$）**：
   $$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
   衡量社群间割裂与闭环程度。
2. **观点极化指数（Polarization Index $\Pi$）**：
   $$\Pi(t) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{b}_i(t) - \bar{\mathbf{b}}(t)\|^2$$
3. **结构洞跨界系数（Effective Size & Bridging Capacity）**：
   衡量 `active_bridger` 是否被物理隔离，系统连通成分（Connected Components）的数量变化。

---

## 3. 系统分层工程架构

```
network_evolve/
├── agent.py       # 智能体信念状态机、异质原型行为决策
├── graph.py       # 动态拓扑图引擎（基于 NetworkX 高效优化）
├── engine.py      # 双向共演调度主循环（离线统计驱动 + 线上 LLM 混合驱动）
└── metrics.py     # 模块度、极化指数、回音室熵等评估度量
```
