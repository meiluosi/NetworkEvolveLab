# 理论形式化推导与数学命题
## Mathematical Formulations, Theorems & Theoretical Bounds

本文件提供 NetworkEvolveLab 复杂网络与大模型认知共演系统的数学理论推导，为学术论文（Methods & Theoretical Analysis）提供可直接引用的形式化证明。

---

### 命题 1：双极隔离网络纽曼模块度的理论数学上界（Upper Bound of Bipolar Modularity）

> **命题 1 (Proposition 1)**: 设时变社交网络 $G=(V, E)$ 演化为 $K$ 个完全互不相通的独立子群（即两两之间无任何交叉边，$E(C_k, C_l) = \emptyset, \forall k \ne l$）。若子群内部连接对称（度分布均等），则纽曼模块度 $Q$ 的严格理论上界为：
> $$Q \le 1 - \frac{1}{K}$$
> 特别地，对于两极对称分离网络（$K=2$ 独立双回音室），模块度数学上界严格满足：
> $$Q \le 0.500$$

#### 形式化证明 (Proof):
根据纽曼模块度（Newman's Modularity）的标准定义：
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
其中 $m = |E|$ 为总连边数，$A_{ij}$ 为邻接矩阵元素，$k_i$ 为节点 $i$ 的度，$c_i \in \{1, \dots, K\}$ 为节点所在社群。

当网络完全分裂为 $K$ 个互不连通的子图 $C_1, C_2, \dots, C_K$ 时，所有连边均位于社群内部。对每一个社群 $C_k$，定义：
- 社群内部边数：$e(C_k) = \frac{1}{2} \sum_{i, j \in C_k} A_{ij}$
- 社群节点度之和：$D(C_k) = \sum_{i \in C_k} k_i$

由于不存在跨群边，必有 $2 \cdot e(C_k) = D(C_k)$，且 $\sum_{k=1}^K e(C_k) = m$。代入模块度公式化简得：
$$Q = \sum_{k=1}^K \left[ \frac{e(C_k)}{m} - \left( \frac{D(C_k)}{2m} \right)^2 \right]$$

令 $a_k = \frac{D(C_k)}{2m} = \frac{e(C_k)}{m}$，显然满足：
$$a_k \ge 0, \quad \sum_{k=1}^K a_k = 1$$

将 $a_k$ 代入模块度表达式：
$$Q = \sum_{k=1}^K (a_k - a_k^2) = \sum_{k=1}^K a_k - \sum_{k=1}^K a_k^2 = 1 - \sum_{k=1}^K a_k^2$$

根据柯西-施瓦茨不等式（Cauchy-Schwarz Inequality）：
$$\sum_{k=1}^K a_k^2 \ge \frac{1}{K} \left( \sum_{k=1}^K a_k \right)^2 = \frac{1}{K}$$
当且仅当 $a_1 = a_2 = \dots = a_K = \frac{1}{K}$ 时等号成立（即各社群规模与度权重完全对称相等）。

因此：
$$Q \le 1 - \frac{1}{K}$$
对于具有左右对称两极阵营的社交网络（$K=2$）：
$$Q \le 1 - \frac{1}{2} = 0.500 \quad \blacksquare$$

#### 科学推论与对假说 H1 的解释：
该定理直接从图论数学上证明了：**在两极对立的社会网络中，无论微观极化多么严重、跨群连边多么彻底被切断，纽曼模块度 $Q$ 都不可能超越 $0.500$**。文献与预注册假说中设定的“$Q > 0.65$”只有在系统涌现出至少 $K \ge 3$ 个密集分离社群（$1 - 1/3 \approx 0.667$）时才在数学上可能。这从根源上确立了假说 H1 证伪的理论自洽性。

---

### 命题 2：多维语义空间下关系非对称撕裂存在性（Asymmetric Link Rupture Theorem）

> **命题 2 (Proposition 2)**: 
> 设一维标量有界信任模型中，两节点认知距离为 $d_{\text{scalar}}(i, j) = |b_i - b_j|$，因满足对称性 $d(i, j) = d(j, i)$，若双方容忍度同质，断连事件（Unfollow）必然成对对称发生。
> 相对地，在多维语义信念空间 $\mathbf{b} \in [-1, +1]^d$（$d \ge 2$）中，若各智能体具有异质的议题显著性权重向量 $\mathbf{s}_i \in \Delta^d$（$\sum_k s_{i,k}=1, s_{i,k} \ge 0$），则断连决策在非零测度空间内具有非对称性：
> $$P(\text{Unfollow}(i \to j) \land \text{Retain}(j \to i)) > 0$$

#### 形式化证明 (Proof):
在多维语义认知交互中，节点 $i$ 认知到邻居 $j$ 的分歧度由加权主观语义距离决定：
$$D_{\mathbf{s}_i}(\mathbf{b}_i, \mathbf{b}_j) = \sum_{k=1}^d s_{i,k} |b_{i,k} - b_{j,k}|$$
同理，节点 $j$ 认知到节点 $i$ 的分歧度为：
$$D_{\mathbf{s}_j}(\mathbf{b}_j, \mathbf{b}_i) = \sum_{k=1}^d s_{j,k} |b_{i,k} - b_{j,k}|$$

取 $d=2$，设节点 $i, j$ 在两个不同议题上的信念差向量为 $\Delta = [|b_{i,1} - b_{j,1}|, |b_{i,2} - b_{j,2}|] = [\delta_1, \delta_2]$。不妨设 $\delta_1 > \theta > \delta_2$（在议题 1 上严重分歧，在议题 2 上高度共识）。

若节点 $i$ 极度重视议题 1（$\mathbf{s}_i = [0.9, 0.1]$），而节点 $j$ 极度重视议题 2（$\mathbf{s}_j = [0.1, 0.9]$）：
- 节点 $i$ 的主观认知失调度：
  $$D_{\mathbf{s}_i} = 0.9 \delta_1 + 0.1 \delta_2 > 0.9 \theta + 0.1 \delta_2 > \theta \implies \text{触发认知失调，边 } (i \to j) \text{ 发生衰减或断连}$$
- 节点 $j$ 的主观认知失调度：
  $$D_{\mathbf{s}_j} = 0.1 \delta_1 + 0.9 \delta_2 < 0.1 \delta_1 + 0.9 \theta < \theta \implies \text{处于可容忍阈值内，边 } (j \to i) \text{ 予以保留并增强}$$

因此，非对称关系断裂集合：
$$\Omega_{\text{asym}} = \{(\mathbf{b}_i, \mathbf{b}_j, \mathbf{s}_i, \mathbf{s}_j) \mid D_{\mathbf{s}_i}(\mathbf{b}_i, \mathbf{b}_j) > \theta \text{ 且 } D_{\mathbf{s}_j}(\mathbf{b}_j, \mathbf{b}_i) \le \theta\}$$
在参数空间中具有严格正测度（$\mu(\Omega_{\text{asym}}) > 0$）。 $\blacksquare$

#### 科学推论与对 RQ1 的解释：
该定理揭示了为什么人类真实大模型交互不会像一维数学玩具模型那样迅速对称瓦解。大模型智能体能够识别“你虽然反对强监管，但赞同开源权重”，从而保留单向关注边，使得全网宏观上依然维持弱连通巨片（Giant Component），微观上涌现出丰富的长尾互嵌结构。

---

### 命题 3：平台去中心化推荐的拓扑渗透与序参量弛豫（Topological Percolation of Diversified Recommendations）

> **命题 3 (Proposition 3)**:
> 在两极极化网络中，引入比例为 $\rho_{\text{div}}$ 的跨阵营异质推荐，跨阵营连边数 $E_{\text{cross}}(t)$ 满足渐近饱和方程：
> $$\frac{d E_{\text{cross}}}{dt} = \rho_{\text{div}} \cdot P_{\text{rec}} \cdot (1 - \phi_{\text{block}}) - \gamma_{\text{decay}} \cdot E_{\text{cross}}$$
> 稳态下跨阵营连边期望为：
> $$E_{\text{cross}}^* = \frac{\rho_{\text{div}} \cdot P_{\text{rec}} \cdot (1 - \phi_{\text{block}})}{\gamma_{\text{decay}}}$$
> 即跨阵营有效连边与推荐比例 $\rho_{\text{div}}$ 呈严格单调一阶线性增长，进而导致纽曼模块度 $Q$ 表现为单调凸函数的剂量响应递减。
