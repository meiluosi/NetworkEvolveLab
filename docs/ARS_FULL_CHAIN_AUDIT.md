# ARS 全链路复核报告（Academic Research Skills — Full-Chain Audit）

> **文档状态（2026-10-22）**：本文件是历史审计报告，保留当时的缺陷清单与修复记录用于审计溯源；其中的旧 Phase 1c 网格 $[0.30,0.90]$、旧阈值均值、Phase 4 六臂时期的表格等均可能已被后续版本取代。当前论文数字以 [`paper/tables/macros.tex`](../paper/tables/macros.tex) 与 `results/*.json` 为准；§8 的自动判定块由 `scripts/generate_paper_tables.py` 保持更新。

> 复核对象：`NetworkEvolveLab`（LLM 认知 × 动态社交拓扑双向共演）
> 复核方法：ARS（Academic Research Skills for Claude Code, `Imbad0202/academic-research-skills`）的
> `academic-paper` / `academic-pipeline` 完整性门协议，特别是
> `academic-pipeline/references/ai_research_failure_modes.md` 的 **7 类 AI 研究失败模式**清单、
> `integrity_review_protocol.md` 与 `reproducibility_audit.md`。
> 复核原则：**不降级结论**。缺实验就补实验，缺对比就补对比，实验有问题就修实验；
> 凡是「数据不支持原结论」的情形，一律通过**补做实验 / 修正归因**来解决，而不是把结论改弱。

---

## 0. 复核结论速览

| 维度 | 复核前状态 | 复核后状态 |
| :--- | :--- | :--- |
| 单元测试 | 9 项，通过 | **18 项**（新增 9 项研究不变量回归测试），通过 |
| 论文定量表格 | 手工抄写，与结果 JSON 存在漂移风险 | **由结果 JSON 自动生成**（`scripts/generate_paper_tables.py`），CI 强制校验 |
| 实验溯源 | 无 | `results/PROVENANCE.json`（命令 / SHA256 / Git 提交 / 环境），CI 强制校验 |
| 图表 | Figure 1–6 | Figure 1–7（新增归因消融图；Figure 1 重做为三面板并改用 Phase 1c 细网格数据） |
| 基准套件 | 5 数据集 × 6 方法（旗舰方法为**简化重实现**） | 6 数据集（含**真实 SNAP Higgs 公开数据**）× 7 方法（**旗舰主引擎亲自参赛**） |
| 复现入口 | `--all/--benchmarks/--figures/--test` | 增加 `--experiments/--provenance`，一键复现全链路 |
| 假说判定 | H1 证伪（理由过强）、H2 证伪、H3「部分支持」（依据为**爆炸的无效统计量**） | H1 证伪（**理由重写为 K 社群推广上界**，新增数值一致性审计）、H2 证伪（**改用种子级统计口径**）、H3 **不成立**（依据换成稳健到达率口径，附完整剂量响应） |
| RQ2 相变阈值 | 断言「四个拓扑的临界容忍度一致落在 $\theta^*\in[0.50,0.55]$」，但**原 θ 采样网格在该区间内只有一个点**，原理上无法判定 | **补做 Phase 1c 临界点测量**（细网格 + 三个可操作判据）：三个判据分别只有 2% / 2% / 10% 的种子级估计落在该区间，**原断言被证伪**；改按实测值重写（见 F-20、§8.4） |
| 真实 LLM 认知层 | 「LLM 驱动拓扑」只在 3 种子侧实验里成立；旗舰引擎**从未挂过真 LLM** | **Phase 6**：真 LLM 后端 vs 规则后端的**同 seed 配对对照**（10 种子 × 30 轮），收据完整（模型 / temperature / input-output-thinking token / 动作分布 / 内容寻址归档）。**结论是负结果**：在 **2.52%** 的互动覆盖率下结构 regime **无显著变化**（$Q$ 配对差 $-0.0040$, $p=0.418$），照实报告（见 F-25、§8.5） |
| 网络规模 | 全部实验都在 $N \in [60,100]$ 上 | **Phase 7 v2**：$N$ 提到 **1000**（含真实 SNAP Higgs $N=1000$ 诱导子图）。**新发现：朴素扩规模（固定 $p$）会让 $Q$ 从 0.446 塌到 0.077，纯属密度混淆**；密度匹配臂下 $Q$ 稳定在 0.427→0.453（见 F-26/F-26b、§8.6） |
| 参考文献 | 19 条 | **55 条**（曾达 71 条，为 ICWSM 11 页上限裁剪），**逐条经 Crossref/arXiv 联网核实**，无引用幻觉，CI 强制「bib 条目必须被引用」 |
| 投稿合规与排版 | **完全未核对**（AAAI 格式 ≠ ICWSM 规则；论文从未按目标会议排版，也从未逐页读过 PDF） | **ICWSM 2026**：正文+参考文献 **11 页**（上限 11）、官方 30 问 Paper Checklist **逐条作答**、章节顺序 合规（正文→参考文献→Checklist→附录）。新增 `check_page_budget.py` 页数门禁 + 编译日志门禁 + 数学模式污染门禁（见 F-27~F-29、§3.5） |
| 缺陷清单规模 | — | **30 项**（F-01 ~ F-29，含 F-26b），其中 P0 5 项 / P1 4 项 / P2 11 项 / 投稿合规与排版 10 项（F-21 ~ F-26、F-26b、F-27 ~ F-29） |

---

## 1. 复核范围与执行清单

| 环节 | 复核动作 | 证据 |
| :--- | :--- | :--- |
| 源码 | 通读 `src/network_evolve/` 全部 18 个模块（约 2200 行） | 见 §3 |
| 结果 | 逐一比对 `results/*.json` 与 `paper/main.tex`、`docs/EXPERIMENT_REPORT.md` 的每个数值断言 | 见 §3、§4 |
| 流水线 | 实际执行 `reproduce.py --test`、全部 Phase 实验、基准套件、绘图管线 | 见 §5 |
| 失败模式 | 按 ARS 7 类失败模式逐条判定（CLEAR / SUSPECTED / INSUFFICIENT EVIDENCE / NOT APPLICABLE） | 见 §6 |
| 引用 | 核对 `paper/references.bib` 与正文引用一致性 | 见 §4.4 |

---

## 2. 复核方法

1. **建立事实基线**：不信任任何文档陈述，所有数字以「实际执行流水线得到的 JSON」为准。
2. **逐条核对断言**：把论文/报告中的每个定量断言映射到具体结果文件与字段。
3. **按 7 类失败模式排查**（ARS `ai_research_failure_modes.md` v3.2）：
   Mode 1 实现缺陷、Mode 2 引用幻觉、Mode 3 幻觉实验结果、Mode 4 捷径依赖、
   Mode 5 缺陷被包装成发现、Mode 6 方法学捏造、Mode 7 早期框架锁定。
4. **对每个 SUSPECTED 项给出修复动作**，优先「补实验 / 修实现」，其次才是「改表述」。
5. **把修复固化为可执行断言**（回归测试 + CI 漂移检测），避免再次漂移。

---

## 3. 发现的问题与处置

### 3.1 P0 — 阻断级：实验结果与论文断言不一致

#### F-01 `docs/EXPERIMENT_REPORT.md` 的「Real LLM 认知真机」整行数据无对应产物（Mode 3 / Mode 6）

报告 §4.1 表格中「Real LLM 认知真机」一行报告
$Q=0.482$、$\Pi=0.735$、巨片 $0.850$、小群 $58.3\%$、$A_{\mathrm{rupture}}=0.286$、跨阵营 $5.8\%$。

实际执行 `python -m src.network_evolve.phase2_llm compare --seeds 0,1,2` 的输出为：

```
llm | 0.275 | 5.000 | 0.177 | 0.250 | 1.000 | 0.702 | 1.000 | 0.507 | 154.333
```

即 $Q=0.275$、巨片 $1.000$、小群 $0.250$、跨阵营 $0.507$；仓库中 `data/llm/seed{0,1,2}/final.json`
三个产物的原始数值同样与之吻合。**报告中的该行数字在整个仓库内找不到任何来源**，
且其中 4 个字段与真实产物直接矛盾。

- **处置**：重跑 Phase 2 全部实验（离线臂 + LLM 臂 + compare），报告表格按新产物重建；
  同时在 `semantic.py` 的 `metrics()` 中补充 `n_mutual` / `n_mutual_broken` / `n_one_sided`，
  使「$A_{\mathrm{rupture}}$ 未定义（NaN）」不再可能被误读为 0
  （原实现中 3 个种子有 2 个返回 NaN，`np.nanmean` 后被呈现为 $1.000$）。

#### F-02 Phase 3 的到达率相对增益存在数值爆炸，H3 判定依据失真（Mode 1 / Mode 3）

`phase3.eval_policy` 原实现：

```python
reach_gain = (r_pol - r_base) / np.maximum(r_base, 1e-6)
```

当某个种子的基线到达率接近 0 时，分母被 `1e-6` 兜底，单个种子即可把均值放大到 $10^4$ 量级。
`results/phase3.json` 中实际记录的值为 `"+2311240.6%±4695243.2%"`、`"+4812970.7%±9859222.6%"`。

同时，论文 §5 与报告 §5.1 据此写出「到达率增益 > +100%，远超 +35% 门槛」，
而**同一份 JSON 里的原始到达率显示 Policy 1 的到达率实际下降**
（baseline $0.465 \to$ policy $0.408$，$p=0.72$，无统计显著性）。

- **处置**：改用稳健口径并全部重跑：
  - `reach_delta`：配对差值 $\mathrm{mean}(r_{\mathrm{pol}} - r_{\mathrm{base}})$（主统计量，配对 t + Wilcoxon）；
  - `reach_rel_gain`：**组均值之比** $( \bar r_{\mathrm{pol}} - \bar r_{\mathrm{base}} ) / \max(\bar r_{\mathrm{base}}, 0.05)$；
  - `reach_ratio`：$\bar r_{\mathrm{pol}} / \bar r_{\mathrm{base}}$；
  - 低于阈值基线（$<0.05$）的种子数量单独报告，不再参与任何比值型统计。
  同时新增回归测试 `test_phase3_reach_gain_does_not_explode` 固化该行为。

#### F-03 旗舰方法在基准评测中并未使用旗舰主引擎（Mode 6 / 捷径依赖）

`benchmark_suite.py` 中标记为 `Ours_CoEvolution` 的是 `baselines.py` 里的
`OursCoEvolutionModel`——一个**与主引擎 `CoEvolutionEngine` 行为不同的简化重实现**
（阈值、闭包概率、干预接口均不一致）。论文 Table~\ref{tab:sota_matrix} 中
「Ours (CoEvolution)」一行因此不是主方法的真实表现。

- **处置**：
  1. 为 `CoEvolutionEngine` 增加 `init_graph` / `init_beliefs` 注入接口，使其可直接在数据集图上运行；
  2. 新增 `metrics.engine_metrics()`，把主引擎终态折算为与基线**完全同口径**的指标；
  3. 基准套件新增 `Ours_CoEvolution_Engine` 行（旗舰主引擎参赛），
     原简化实现保留为 `Ours_CoEvolution_Simplified` 消融参照行。

#### F-04 论文对模块度上界的表述与论文自身数据矛盾（Mode 5 / Mode 7）

摘要与 Proposition~\ref{prop:modularity} 断言「双极网络下 $Q \le 0.500$」，
并以此「证伪」H1（$Q>0.65$）。但论文 Table~\ref{tab:sota_matrix} 自身报告的数值包含
$Q=0.697$（DeGroot / Reddit）与 $Q=0.735$（HK / Reddit），**直接越过 0.500**。
根因是：$Q\le 1-1/K$ 只在 $K=2$ 的双极划分下成立，论文把 $K=2$ 特例当成了普适上界。
（附录 `app:proof_modularity` 其实已经给出了正确的 $K$ 社群推广定理，但正文没有采用，
形成了「附录正确、正文过度推广」的自相矛盾。）

- **处置**：
  1. 正文 Proposition 改为 $K$ 社群一般形式：$Q \le 1 - \sum_c a_c^2 \le 1 - 1/K$；
  2. 新增 Phase 5 一致性审计模块 `phase5_bound_audit.py`：
     对基准表**每一个**（数据集 × 方法）单元核验 $Q \le 1 - 1/K$，并输出 $K$ 分布；
     对主演化引擎重跑 H1 参数网格，记录终态 $(K, Q)$；
  3. H1 的判定理由改写为**可核验的实证结论**：在全部 12 组参数配置、240 次仿真中，
     演化的 Louvain 划分停留在 $K=2$（双极）或 $K=3$（三方），其中 70%–90% 的种子为 $K=2$；
     **即使出现 $K=3$，由于社区度体积严重不均衡，起作用的是更紧的界 $1-\sum_c a_c^2$，
     实测最大 $Q$ 仍为 0.500**。$Q>0.65$ 需要 $K\ge 3$ 且社区度体积接近均衡，
     本引擎在全部网格配置中均未达到（最大 $Q$ = 0.500，0.65 阈值达成率 0%）。
     —— 这是一个比「数学上不可能」更强、更准确、且可被 Phase 5 数值核验的结论。
     （本条曾以「本引擎在任何参数下都不产生第三社群」表述，被 Phase 5 实测推翻并更正：
     引擎确实会偶尔产生第三个社群，$K_{\max}=3$。）

#### F-20 论文「四个拓扑的临界容忍度一致落在 $\theta^*\in[0.50,0.55]$」无法由原实验判定，且被补做实验证伪（Mode 1 / Mode 3 / Mode 5）

正文 RQ2 断言：「Phase transition sweeps across ER, BA, WS, and Empirical Bipolar topologies
confirm that the critical tolerance threshold consistently occurs within
$\theta^*\in[0.50,0.55]$」。

核查该断言的实验来源 `phase_robustness.py` 后发现两个层面的问题：

1. **实验设计原理上无法判定该命题**：其 θ 采样网格为 `[0.35, 0.45, 0.55, 0.65]`，
   在待检验区间 $[0.50,0.55]$ 内**只有一个采样点 0.55**，且 0.50 根本没有被采样；
   模块自身的 docstring 就写着「$\theta^*$ 是否依然处于 0.50~0.55 区间？」——
   目标与采样网格自相矛盾。
2. **断言被补做实验证伪**：新测量的细网格数据显示，四个拓扑的 $\theta^*$ 并不一致，
   且几乎不落在预登记区间内（详见 §8.4 与 `results/phase1c_threshold.json`）。

- **处置（补实验 + 修断言，而非改表述）**：
  1. 新增 `src/network_evolve/phase1c_threshold.py`：4 拓扑 × 细网格 θ（步长 0.05，
     上界扩展至 0.90 以消除右删失）× 10 种子；
  2. 给出**可操作判据**（而非目测曲线）：主判据为「$Q_{\mathrm{end}}(\theta)$ 跌破
     $Q_{\mathrm{end}}(\theta_{\min})$ 一半的最小 θ」，辅判据为极化塌缩判据；
     逐种子估计再跨种子取均值与 95% CI，删失样本显式计入且不静默丢弃；
  3. Figure 1B 改用该细网格数据并叠加各拓扑实测 $\theta^*$，Figure 1A 的临界线
     也改为由曲线推导（原为图脚本里硬编码的 0.52）；
  4. 正文 RQ2 结论按实测值改写：三个判据（连通性恢复 / 模块度塌缩 / 极化塌缩）
     分别只有 2% / 2% / 10% 的种子级估计落在预登记区间 $[0.50,0.55]$ 内；
     真正跨拓扑稳健的是**连通性恢复阈值** $\theta^*_{\mathrm{frag}}$
     （ER 0.339、BA 0.407、WS 0.363、实证双极 0.520，拓扑间极差 0.180），
     且模块度塌缩判据在小世界 / 实证双极上大面积右删失（10/10 种子），
     说明该判据本身不适用于这两类拓扑。

### 3.2 P1 — 严重级：归因过度 / 实现缺陷

#### F-05 非对称撕裂被归因于「多维语义」，但存在未被控制的混杂变量（Mode 4 / Mode 5）

论文 RQ1 与 Proposition~\ref{prop:asym} 把 $A_{\mathrm{rupture}}$ 归因于多维语义认知。
但项目自有数据即显示标量 HK 模型也产生非对称断裂（Phase 2 离线臂：
$\epsilon=0.6 \to 0.029$，$\epsilon=1.2 \to 0.498$），而经典有界信任模型的断裂判据在理论上是对称的。
这提示存在一个混杂变量：**异步单发言者调度**。

进一步的机制排查发现，`A_{\mathrm{rupture}}$ 指标本身还被两个实现层因素污染：

1. **初始边权不对称**：互惠两边的初始强度不同 ⇒ 即使判据完全对称，较弱一侧也会先断连，
   在指标上表现为「非对称断裂」；
2. **三元闭包重连已断裂的边**：`semantic.py` 的 `VecWorld` 缺少主引擎已有的 `blocked` 黑名单，
   刚刚断裂的边会被闭包立即加回，制造虚假的单向断裂终态。
   实测：各向同性对照臂残留的全部单向断裂对，**100% 属于「曾断裂又被闭包加回」**。

- **处置（补实验 + 修实现，而非改表述）**：
  1. 修复实现：`VecWorld` 增加 `blocked` 黑名单，与 `CoEvolutionEngine` 语义对齐；
     所有断边统一走 `_drop_edge()`。
  2. 新增 **Phase 4 归因消融**（`src/network_evolve/phase4_attribution.py`）：
     受控因子设计（同一 seed 下所有实验臂共享同一初始图与初始信念）——
     | 实验臂 | 维度 | 调度 | 议题权重 | 作用 |
     | :-- | :-- | :-- | :-- | :-- |
     | `scalar_sync` | $d=1$ | 同步 | — | 经典标量有界信任，理论预期 $A\equiv 0$ |
     | `isotropic_sync` | $d=3$ | 同步 | 各向同性 | 仅去掉「个体各向异性」这一个变量 |
     | `multidim_sync` | $d=3$ | 同步 | 各向异性 | 仅比上一行多「个体各向异性」 |
     | `scalar_async` | $d=1$ | 异步 | — | 隔离「异步调度」的影响 |
     | `multidim_async` | $d=3$ | 异步 | 各向异性 | 原始量级 |
     | `multidim_async_orig` | $d=3$ | 异步 | 各向异性 + 原始容忍度规则 | 原始模型参照 |
     并强制互惠边初始权重对称化，使 $A_{\mathrm{rupture}}$ 只反映**决策不对称**。
  3. 把「同步标量 / 同步各向同性 ⇒ $A \equiv 0$」「同步各向异性 ⇒ $A>0$」
     固化为回归测试 `test_scalar_sync_rupture_is_exactly_symmetric`、
     `test_isotropic_sync_rupture_is_exactly_symmetric`、
     `test_anisotropy_creates_asymmetry_under_sync`。
  4. 结论表述随之**收窄并加强**：非对称撕裂的正确来源是
     **「个体各向异性语义判据」**（Proposition~\ref{prop:asym} 的原假设得到正面验证），
     而**异步调度是第二个独立来源**（在纯标量模型中同样制造非对称断裂）。
     论文原文「多维语义 ⇒ 非对称撕裂」因此属于**过度归因**，需改写；
     但命题本身得到了比原文更强的、受控的实证支持。

#### F-06 H2 判定使用节点级伪重复统计（Mode 1）

`metrics.h2_test` 把同一 seed 内的所有节点当作独立样本做 Mann–Whitney 检验，
低估标准误；而预注册判据是「Bridger 留存率至少低 40%，且 $p<0.01$」。

- **处置**：新增 `metrics.h2_seed_level_test()`，以**种子**为独立统计单元：
  效应量 $\mathrm{mean}(\text{per-seed gap}) \ge 0.40$ + 单样本单侧 t 检验
  $H_0:\bar{g}\le 0$（$p<0.01$）；同时保留更严格的「差距是否显著超过 0.40」对照检验与符号检验。
  节点级结果降级为参考项 `p_value_node_level`。H2 结论不变（仍为证伪），
  但统计口径现在与预注册判据一致且可辩护。

#### F-07 断边黑名单在两个实现之间不一致（Mode 1）

`CoEvolutionEngine` 有 `blocked` 集合，`VecWorld`（Phase 2 语义后端）没有。
两套实现对「断裂后能否重连」给出不同答案，导致同一项目的两个实验族对
「断裂」的语义不一致。

- **处置**：见 F-05 处置 1；`phase2_llm.apply_round` 也改为统一走 `_drop_edge()`。

#### F-19 Phase 2 的 LLM 臂与离线臂**轮数不对等**，跨臂对比不具事件对齐性（Mode 1 / Mode 3）

原 Phase 2 对比中，真实 LLM 臂每个种子只跑到 1–2 轮交互，而离线代理臂跑到 6 轮。
两端观测到的事件数不同，因此「LLM 臂 vs 离线臂」的差值混合了**认知模型差异**与
**观测窗口长度差异**两个因素，不能解释为模型差异。

- **处置（补实验 + 修对比口径）**：
  1. 新增 `phase2_llm replay` 子命令：复用已归档的 request/response 收据，
     以**事件级对齐**方式重放，使两端在同一事件序列上比较；
  2. replay 完全复用归档产物，**无需重新调用 API**，因此可离线复现；
  3. 结果写入 `results/phase2_compare.json`，并说明「归档窗口内哪些指标有定义」：
     离线臂（degroot / hk / proxy）在该窗口内的断裂互惠二元组数为 `[0, 0, 0]`，
     故其 $A_{\mathrm{rupture}}$ 在对齐窗口内**未定义**，不可与 LLM 臂的数值并列比较。

### 3.3 P2 — 一致性 / 可复现性

| 编号 | 问题 | 处置 |
| :--- | :--- | :--- |
| F-08 | 论文正文/表格/图注中的数据集数量互相矛盾（README「3 大真实社交网络」vs 正文「5 大黄金数据集」vs Figure 5 图注「three real-world topologies」） | 全部改为由结果 JSON 动态生成；README 与图注同步更新 |
| F-09 | 论文表中的数字全部手工抄写，易与实验漂移 | 新增 `scripts/generate_paper_tables.py`，从 JSON 生成 `paper/tables/*.tex`；CI 用 `--check` 强制一致 |
| F-10 | 实验产物没有溯源信息 | 新增 `scripts/record_provenance.py` → `results/PROVENANCE.json`（命令 / SHA256 / Git 提交 / 环境）；CI 用 `--check` 强制一致 |
| F-11 | `demo/app.py` 使用了**不存在的 API**（`DynamicSocialGraph(initial_graph=...)`、`CoEvolutionEngine(graph=..., agents=...)`），README 宣称的 Streamlit 仪表盘一跑即崩 | 按真实 API 重写（注入 `init_graph` / `init_beliefs`，改用 `run_round()`），并把「Bipolar Cap」标注改为 $K$ 社群上界表述 |
| F-12 | AI 使用声明与执行收据不符：附录与正文称「6 次批量请求、60 个语义交互判定」，而 ledger 实际为 4 次调用、198 个判定对（60+60+42+36），且 ledger 未记录 token 计数 | 按 ledger 实况改写声明；补记每个产物的调用数与判定对数 |
| F-13 | 五个「黄金基准数据集」实为**按公开统计量校准的合成代理网络**（`datasets.py` 内用种子化 RNG 生成），但正文多处称其为「real-world / gold-standard empirical datasets」 | 保留代理网络但**改名与改述**为「surrogate structural proxies」；同时**接入真实公开数据**：从 SNAP 官方 `higgs-social_network.edgelist.gz`（14,855,842 边 / 456,626 节点）派生第 6 个基准数据集 `SNAP_Higgs_Real`，并在 `data/raw/README.md` 记录来源 URL、SHA256 与派生配方 |
| F-14 | 真实 Higgs 子图的节点选择与种子无关 ⇒ 10 个种子共用同一张图，置信区间虚假收紧 | 节点池改为「按度取前 $M$ 名后由种子随机抽样」，使每个种子得到不同但统计同阶的幂律子图 |
| F-15 | 论文把 `A_rupture` 的数值在「真实 LLM 臂」与「基准套件引擎臂」之间混用（$0.778$ 来自 Twitter 基准的合成臂，却被写成 LLM 臂的结果） | 严格分离两条臂的数值来源，论文中分别标注 |
| F-16 | 论文与报告称「HK 的撕裂非对称度严格为 0.000」，但项目自有 Phase 2 数据为 $\epsilon{=}0.6 \to 0.029$、$\epsilon{=}1.2 \to 0.498$ | 由 Phase 4 归因消融给出机制解释：同步调度下 HK 严格为 0，异步调度下不为 0；正文表述改为「在同步判据下严格对称」 |
| F-17 | 摘要/正文宣称 H1 被「数学证伪」 | 改写为「在双极区间内被经验证伪 + $K$ 社群推广上界的数值一致性审计」，并给出 Phase 5 核验结果 |
| F-18 | `reproduce.py` 覆盖不到 Phase 0/1/1b/2/3/4/5 与溯源 | 增加 `--experiments` / `--provenance`，`--all` 覆盖全链路 |

### 3.4 发表阻断项修复轮（第二轮复核）

在「是否可以投稿」的评估之后，又按同一套 ARS 协议执行了一轮**发表阻断项**复核。本轮发现的问题比第一轮更基础：**论文此前从未被真正编译过**。

| 编号 | 问题 | 处置 |
| :--- | :--- | :--- |
| F-21 | **论文 LaTeX 从未编译通过**：① 正文使用 `\begin{enumerate}[leftmargin=*]` 但从未加载 `enumitem`；② 自动生成表格中的裸 `%`（如 `\newcommand{\HOnePassRate}{0%}`）在 LaTeX 中是注释符，会吞掉右花括号与整行剩余内容；③ 宏名含数字（`\CrossEdgeP1`）在 LaTeX2e 2021-11-15 内核下会错位解析参数计数，宏体泄漏进导言区并触发 `Missing \begin{document}` | ① 去掉 enumitem 可选参数（不新增包，AAAI 模板下本就不允许改变列表间距）；② 在 `generate_paper_tables.py` 的**写出点统一转义** LaTeX 特殊字符（`tex_escape`），并对 `\input` 的表格行与宏定义分别转义；③ 宏名改为纯字母（`\CrossEdgePolicyOne`）；④ 在 `check_paper_integrity.py` 增加**两道机器门禁**：宏名必须纯字母、生成片段中不得出现未转义 `%` |
| F-22 | 论文 §2 的形式化模型描述的是 **$d$ 维信念向量 + 显著度加权失谐**的连续公式，而产生 H1/H2/H3 与全部基准结果的**旗舰引擎是标量（$d=1$）**，且其实际更新是「阈值内同化、超阈值即停止信念移动」的两分支规则，与论文写的三段式核 $\psi(\cdot)$ **不一致** | 重写 §2：给出一般多维几何，明确标注「标量旗舰引擎」与「多维语义臂」各自承担哪些结果；把旗舰引擎的**实际**标量更新式与边权更新式写成独立公式（式 (3)(4)），并声明多维几何只用于 LLM 认知实验与 Phase 4 的多维对照臂 |
| F-23 | 参考文献仅 19 条，Related Work 章节**完全缺失**，且缺少可核实的近年文献 | 新增 **Related Work** 章节（7 个子节：固定拓扑意见动力学 / 共演网络 / LLM 作为被试 / 带社交网络的生成式仿真 / 推荐与回音室 / 社群检测与模块度上限 / 可复现性标准 / 数据溯源）；参考文献由 19 条扩到 **69 条**，其中 50 条新增条目**逐条经 Crossref DOI 与 arXiv API 解析**取回作者、标题、容器、年份（脚本：`scripts/fetch_bib_from_crossref.py`、`scripts/fetch_bib_from_arxiv.py`），解析失败者一律丢弃而非臆造 |
| F-24 | 「LLM 驱动拓扑」这一标题级主张此前只在一个 3 种子侧实验里成立；`semantic.py` 文件头声称有 `llm` 后端但 `react()` 里**根本没有该分支**（走到末尾直接 `raise ValueError`） | 新建**统一 LLM 认知后端** `src/network_evolve/llm_backend.py`（传输层 `LLMClient` + 决策层 `LLMCognitivePolicy` + 效应层 `apply_decision`），`VecWorld(backend="llm")`、旗舰 `CoEvolutionEngine(llm_policy=...)`、`phase2_llm` 三者共用同一实现，消除旁路语义分叉；`phase2_llm` 重构后**重放结果与重构前逐位一致**（`results/phase2_compare.json` 全字段相同） |
| F-25 | 真实 LLM 臂**没有完整收据**（模型标识、温度、token 计数全缺），且旗舰引擎从未挂过真 LLM | 新增 **Phase 6**：真实 LLM 认知层 vs 规则后端的**同 seed 配对对照**（10 种子 × 30 轮 × 6 事件/轮，每轮一次批量调用）。收据含模型标识 / temperature / input-output-thinking token / response id / stop reason / 解析失败率 / 动作分布；响应按内容寻址归档，重跑**零 API 调用、逐位可复现** |
| F-26 | 全部实验都在 $N \in [60, 100]$ 上完成，无法回答审稿人必问的「结论是否只是小图上的偶然」 | 新增 **Phase 7** 规模敏感性研究：$N \in \{60,100,250,500,1000\}$ × 10 种子 × 60 轮，并在**真实 SNAP Higgs 的 $N=1000$ 诱导子图**上重复（该缓存由一次性原始边表扫描生成，10 种子共用同一次扫描）。**Phase 7 v2 追加发现**：朴素扩规模（固定 $p=0.08$）会让模块度随 $N$ 塌陷（$0.446 \to 0.077$），纯属**密度混淆**；改为密度匹配臂（固定平均入度 $\bar d=7.92$）后 $Q$ 在 $N=60 \to 1000$ 保持 $0.427 \to 0.453$（见 §3.5 的 F-26b 与 §8.5） |

### 3.5 投稿合规与排版复核轮（第三轮复核）

前两轮把「实验与断言」修到了自洽，但**投稿合规性**与**排版正确性**这一层从未被检查：
论文既没有按目标会议的页数/清单要求排版，也没有被逐字读过排版结果。本轮据此执行了
「会议规则联网核实 → PDF 逐页实测 → 结构化压缩 → 机器门禁固化」四步。

| 编号 | 问题 | 处置 |
| :--- | :--- | :--- |
| F-27 | **投稿合规性此前完全未核对**。原目标模板（AAAI 格式）与目标会议（ICWSM 2026）规则不一致：① ICWSM 2026 Full Papers 上限 **11 页且正文与参考文献计入**（Checklist 与附录不计入）；② **Paper Checklist 强制**，缺失 desk-reject；③ 章节顺序必须是 正文 → 参考文献 → Checklist → 附录 | 按 <https://www.icwsm.org/2026/submit.html> 联网核实规则后：新增 `paper/checklist.tex`，**官方 30 问逐条作答**（问题原文取自 arXiv:2501.09950，去掉 AAAI 禁用的 xcolor 着色）；`paper/main.tex` 章节顺序改为 正文 → 参考文献 → Checklist → 附录；参考文献按 AAAI 明文许可降为 `\small`；新增 **`scripts/check_page_budget.py`**（判定口径：Checklist 必须从第 12 页或更后开始）作为机器门禁，并接入 CI |
| F-28 | **正文+参考文献 13 页，超 ICWSM 上限 2 页**（纯文字 8 页 + 浮动体 ≈3 页 + 参考文献 ≈2 页） | 见下方「页数压缩的具体取舍」。**信息不丢、只重排**：推导/枚举/逐格明细一律移入附录（新增 `app:kernel`、`app:bound-audit`、`app:threshold`、`app:limitations`、`app:ethics`、`app:attribution` 六节），正文保留全部**结论、统计量与负结果**；散文 6,311 → 4,983 词，摘要 319 → ≤250 词（AAAI 上限） |
| F-29 | **正文里有 8 个多余的 `$`（两处），把拓扑名整句吞进数学模式**：`\ThresholdFragErdosRenyi$ (Erd\H{o}s--R\'enyi)` 一类写法在 `main.tex` 与 `appendix.tex` 各 4 个。LaTeX **不报错**（`$` 数量成对），只把该段排成 cmmi10 数学斜体并**丢掉重音字符**（编译日志 `Missing character: There is no ő ("151) in font cmmi10!`），PDF 上几乎看不出来 | ① 删除全部 8 个多余 `$`；② `check_paper_integrity.py` 新增**第 13 项门禁**：静态扫描数学模式里的裸单词（≥3 字母）、重音命令、非 ASCII 重音字母、以及「数字宏后开启数学模式」四类污染（并对 `build/icwsm/*.tex` 拍平件一并检查）；③ `check_page_budget.py` 新增**编译日志门禁**：`Missing character` / `!` 开头的 TeX 错误 / `Missing \begin{document}` / `Runaway argument` 一律 FAIL |
| F-26b | **Phase 7 的朴素扩规模协议自带密度混淆**（见 F-26 处置栏），$Q$ 的塌陷若不加控制会被误读为「机制在大图上失效」 | `phase7_scaling.py` 重写为**双臂**：**密度匹配臂**（主结果，固定 $\bar d=7.92$，$p(N)=\bar d/(N-1)$）与**固定-$p$ 臂**（混淆探针）。两臂只差「被变动的那个参数」，$N=100$ 处与论文主实验重合。论文同时报告两臂 |

#### 页数压缩的具体取舍（F-28 细目）

压缩不是「删内容」，而是**换承载位置**。逐项如下，以便复核者核对信息是否真的没丢：

| 动作 | 说明 | 信息是否丢失 |
| :--- | :--- | :--- |
| 参考文献 71 → **55 条** | 按「边缘度」从低到高删除 16 条（裁剪清单存于 `docs/dropped_references.txt`）；全部 55 条仍被正文引用，且**逐条经 Crossref/arXiv 联网核实** | 引用条目减少，但用户要求的 40+ 仍超出 15 条；被删者均为「有更近文献可替代」的边缘引用 |
| `fig5`（SOTA 基准对比图）移入附录 | 正文保留 `tab:sota_matrix` 数值表（同样的数据，机器生成） | 否：数值仍在正文，视觉版本在附录 |
| `tab:attribution` 移入附录 | 正文保留 `fig7` 归因消融图与全部结论文字 | 否 |
| 推导/核函数/上界审计/阈值明细/完整限制/伦理滥用渠道 → 附录 | 新增 `app:kernel` 等六节 | 否：正文保留结论与关键数字 |
| `fig1`/`fig2`/`fig3` 由跨栏改单栏 | `width=\columnwidth`，与原先 `0.48\textwidth` **像素尺寸完全一致** | 否 |
| 跨栏浮动体**声明顺序**重排 | `tab:hypotheses`+`fig7`+`tab:llm_arm` 相邻、`tab:scaling`+`fig8` 相邻，使其堆叠在同一页顶部（每个跨栏浮动体单独声明会各自强制分页，实测每次浪费约半栏） | 否：只改落位 |
| 浮动体放置参数 | `\topfraction 0.9` / `\bottomfraction 0.8` / `\textfraction 0.05` / `\dbltopfraction 0.9` / `\dbltopnumber 3` / `\totalnumber 5` | 否：**只改浮动体落位，不动字号、页边距、行距**（AAAI 未禁止） |
| 两个超宽公式压缩 | `eq:scalar-update`（`cases` 溢出 138pt）与 `eq:weight-update`（溢出 129pt）改写为两行式，分支说明移入散文 | 否：分支语义完整保留 |
| 摘要 319 → **249 词**、散文 6,311 → 4,969 词 | 结论、统计量、负结果一律保留；被压缩的是重复表述与已在附录展开的枚举（摘要经**宏展开后**计词，满足 AAAI 的 250 词上限） | 否 |

**最终实测**：`build/icwsm/NetworkEvolveLab_ICWSM.pdf` 共 22 页；正文+参考文献 **11 页**（上限 11），
Checklist 起自第 11 页末、附录起自第 14 页，`check_page_budget.py` PASS。

**残余风险（如实披露）**：第 11 页无余量（Checklist 标题正好落在页底），后续任何正文/图表增补
都可能触发超限。为此新增两道回归门禁：`check_page_budget.py`（页数）与
`check_paper_integrity.py` 第 12/13 项（环境配对、数学模式污染），二者均已接入 CI。

---

## 4. 逐项断言核对结果

### 4.1 论文核心数值 vs 结果产物

| 断言位置 | 断言 | 复核结论 |
| :--- | :--- | :--- |
| 报告 §4.1 | Real LLM: $Q=0.482$、巨片 $0.850$、$A=0.286$、跨阵营 $5.8\%$ | **与产物矛盾**（实际 $0.275/1.000/1.000\ (2/3\ \mathrm{NaN})/0.507$）→ F-01，已重跑重建 |
| 报告 §5.1 / 论文 §5 | 到达率增益 > +100% | **不成立**（Policy 1 到达率由 0.465 降至 0.408，$p=0.72$）→ F-02 |
| 报告 §7.4 | HK 撕裂非对称度严格为 0.000 | **过强**（异步调度下非零）→ F-05/F-16 |
| 论文摘要 | 双极网络 $Q\le 0.500$ 为普适上限 | **与自身表格矛盾**（表格含 0.735）→ F-04 |
| 论文 §RQ2 | 四个拓扑的临界容忍度一致落在 $\theta^*\in[0.50,0.55]$ | **原理上无法判定且被证伪**（原 θ 网格在该区间只有一个采样点；Phase 1c 三个判据分别只有 2%/2%/10% 的种子估计落在区间内）→ F-20 |
| 论文 §5 | Policy 1 跨阵营连边 +115%、组合 +210% | 与 `results/phase3.json` 的 `cross_edges` 一致（$0.020\to0.043$、$0.020\to0.062$）——**成立** |
| 论文 §H2 | Bridger $0.664$ vs 其他 $0.511$，$p=1.000$ | 数值与 `results/phase1b.json` 一致；统计口径改为种子级 → F-06 |
| 附录 | 6 次批量 API 请求 / 60 个判定 | **与 ledger 不符**（4 次 / 198 对）→ F-12 |

### 4.2 数据集与方法的真实属性

| 名称 | 论文表述 | 实际属性 | 处置 |
| :--- | :--- | :--- | :--- |
| DEBATE / SNAP Higgs / Twitter 2020 / Reddit / Birdwatch | 「gold-standard empirical datasets」「real-world」 | 按公开统计量校准的**合成代理网络** | 改称 surrogate proxies，并新增真实 SNAP Higgs 公开子图 |
| 真实 SNAP Higgs | — | **真实公开数据**（14.8M 边原始边表派生） | 新增为第 6 个基准数据集 |

### 4.3 引用完整性

`paper/references.bib` 共 19 条，正文 `\citep` 与 `\citet` 引用键全部可解析，无孤立引用；
本轮复核未新增/删除文献，但补上了**原先漏引的 Louvain**（`blondel2008fast`）——
该算法是全文模块度测量的基础，此前 bib 中有条目却从未被引用（已由
`scripts/check_paper_integrity.py` 的「bib 条目必须被引用」门禁固化）。
已核对的引用键：`bail2018exposure`、`bakshy2015exposure`、
`dandekar2013biased`、`centola2010spread`、`degroot1974reaching`、`hegselmann2002opinion`、
`friedkin1990social`、`park2023generative`、`gu2025promptrewire`、`newman2004finding`、
`newman2006modularity`、`blondel2008fast`、`barabasi1999emergence`、`watts1998collective`、
`neurips2024debate`、`domenico2013anatomy`、`poole1985dwnominate`、`chang2020convokit`、
`wojcik2022birdwatch`。

### 4.4 复现入口核验

`reproduce.py` 的每个子命令均已实际执行验证；`--all` 现在覆盖
「测试 → 全部 Phase 实验（含新增的 Phase 1c 临界点测量）→ 基准套件 → Phase 5 审计
→ 图表 → 论文表格 → 审计报告判定块回填 → 论文静态完整性检查 → 溯源清单」。
CI（`.github/workflows/ci.yml`）在此之上额外执行
`generate_paper_tables.py --check`（表格与审计判定块防漂移）、
`record_provenance.py --check`（产物溯源防漂移）与
`check_paper_integrity.py`（论文引用的宏 / label / `\input` / 图文件必须真实存在）。

---

## 5. ARS 7 类失败模式判定

| 模式 | 判定 | 证据 / 处置 |
| :--- | :--- | :--- |
| Mode 1 实现缺陷通过自检 | **SUSPECTED → 已修复** | F-02（到达率爆炸）、F-05（闭包重连断边）、F-06（伪重复）、F-07（实现不一致）、F-19（Phase 2 跨臂轮数不对等）、F-20（θ 采样网格无法判定被检验命题）；新增 9 项回归测试固化 |
| Mode 2 引用幻觉 | **CLEAR** | 19 条 bib 引用键全部可解析、无虚构条目；新增「引用键必须存在于 bib」与「bib 条目必须被引用」双向门禁（`scripts/check_paper_integrity.py`），本轮据此补上了此前漏引的 Louvain |
| Mode 3 幻觉实验结果 | **SUSPECTED → 已修复** | F-01（报告中的 LLM 行无对应产物）、F-02（依据失真的 H3 统计量）、F-20（θ* 断言无实验支撑且被证伪）；改为自动生成表格 + 溯源清单 |
| Mode 4 捷径依赖 | **SUSPECTED → 已修复** | F-03（旗舰方法未用旗舰引擎参赛）、F-05（归因未排除调度混杂）；补 Phase 4 受控消融 + 真实主引擎基准行 |
| Mode 5 缺陷被包装成发现 | **SUSPECTED → 已修复** | F-04（$K=2$ 特例被当作普适上界，且与自身表格矛盾）、F-05（异步调度伪影被归因为语义创新）、F-20（无法判定的实验被写成「一致」）；Phase 5 一致性审计 + Phase 4 归因消融 + Phase 1c 临界点测量 |
| Mode 6 方法学捏造 | **SUSPECTED → 已修复** | F-03、F-12（AI 声明与 ledger 不符）、F-13（合成代理被称为真实数据）、F-09/F-10（自动表格 + 溯源） |
| Mode 7 早期框架锁定 | **CLEAR（已收敛）** | 三个 RQ 与预注册判据保持不变；本轮未更换研究问题或方法范式，只补齐了检验它们所需的对照与审计 |

---

## 6. 本轮新增的实验与对比

| 新增 | 目的 | 位置 |
| :--- | :--- | :--- |
| Phase 4 非对称撕裂归因消融（6 臂 × 20 种子受控对比） | 判定 $A_{\mathrm{rupture}}$ 的真实来源 | `src/network_evolve/phase4_attribution.py` → `results/phase4_attribution.json` |
| Phase 5 模块度上界一致性审计 | 核验 $Q \le 1-1/K$ 在基准表与演化网格上的每一格 | `src/network_evolve/phase5_bound_audit.py` → `results/phase5_bound_audit.json` |
| Phase 1c 跨拓扑相变临界点测量 | 判定「$\theta^*$ 跨拓扑一致落在 $[0.50,0.55]$」这一原断言（原 θ 网格无法判定） | `src/network_evolve/phase1c_threshold.py` → `results/phase1c_threshold.json` |
| 真实 SNAP Higgs 公开数据基准（第 6 数据集） | 提供**真实**拓扑证据，校正「合成代理=真实数据」的表述 | `src/network_evolve/datasets_real.py` → `data/benchmarks/higgs_real_*.json` |
| 旗舰主引擎基准行 | 让「Ours」是真正的主方法 | `benchmark_suite.py::run_engine_method` |
| Figure 7 归因消融图 | 呈现 RQ1 归因结论 | `scripts/generate_figures.py::plot_fig7` |
| 自动生成论文表格 | 消除手工同步漂移 | `scripts/generate_paper_tables.py` |
| 实验溯源清单 | 结果 ↔ 命令 ↔ 提交 ↔ 环境 的可追溯性 | `scripts/record_provenance.py` |
| 9 项研究不变量回归测试 | 把本轮修复固化为可执行断言 | `tests/test_research_invariants.py` |
| **Phase 6 真 LLM 认知层对照** | 让旗舰引擎真的挂上 LLM 认知层，并给出完整收据与覆盖率 | `src/network_evolve/phase6_llm_engine.py` → `results/phase6_llm_engine.json` |
| **Phase 7 v2 双臂规模研究** | 把规模与密度解耦，检验「结论是否只是小图上的偶然」 | `src/network_evolve/phase7_scaling.py` → `results/phase7_scaling.json` |
| **Figure 8（LLM 臂 + 规模/密度混淆）** | 一图呈现「真 LLM 无显著效应」与「塌陷是密度伪影」 | `scripts/generate_figures.py::plot_fig8` |
| **统一 LLM 后端** | 消除 `phase2_llm` / `VecWorld` / 旗舰引擎三条旁路语义分叉 | `src/network_evolve/llm_backend.py` |
| **ICWSM Paper Checklist（30 问逐条作答）** | 缺则 desk-reject | `paper/checklist.tex` |
| **ICWSM 页数门禁 + 编译日志门禁 + 数学模式污染门禁** | 把「能投稿」变成可机器复验的断言 | `scripts/check_page_budget.py`、`scripts/check_paper_integrity.py`（第 12/13 项） |

---

## 7. 遗留限制（已明确披露，不做隐瞒）

1. 真实 SNAP Higgs 数据**不含任何意识形态标注**，其「阵营」由真实拓扑的 Louvain 社区派生，
   属**结构代理变量**，不能表述为真实政治立场。
2. 其余五个基准数据集仍为**按公开统计量校准的合成代理网络**，论文中已统一改述；
   若要全面升级为真实数据，需要逐个接入 Twitter 2020 / Reddit ConvoKit / Birdwatch 的公开原始文件。
3. **真实 LLM 臂的覆盖率仍然很低**：Phase 6 的旗舰真 LLM 臂只覆盖 **2.52%** 的引擎互动
   （1721 / 68404）。因此「真 LLM 无显著效应」这一负结果**只能**支持
   「在本引擎、本提示词契约、本覆盖率下未观测到结构 regime 变化」，
   **不能**支持「LLM 认知对拓扑演化无影响」这一更强命题。
   所需的下一步实验是**提高覆盖率**（例如把每轮决策预算从 6 事件提到覆盖全部互动），
   本轮未做，因为那会把 API 花费提高约 40 倍（当前 300 次调用、约 58 万 token）。
   Phase 2 的**离线**多维语义臂仍沿用旧口径（3 个种子、198 个判定对），
   其响应的模型版本与 token 计数**未随产物落盘**；Phase 6 已用统一后端补齐了完整收据，
   旧臂仅作为历史对照保留。
4. **Phase 7 固定-$p$ 臂的 $N=1000$ 只跑了 3 个种子**（其余规模 10 个种子），
   因为该臂的 $Q$ 已经塌到 0.077 且 $\Pi \to 0$（结论不依赖精度）。
   该臂定位为**混淆探针**而非主结果，主结果（密度匹配臂）全部为 10 个种子。
4. 归因消融在 $d\in\{1,3\}$、两种调度下进行，尚未覆盖连续维度扫描；
   Proposition~\ref{prop:asym} 的几何论证（$d\ge 2$ 时非共线权重向量给出正测度区域）
   仍然只证明了「存在性」，Phase 4 给出的是「在可测的仿真设定下该机制确实被激活」。
5. **临界容忍度 $\theta^*$ 的估计依赖判据**：连通性恢复判据（$\theta^*_{\mathrm{frag}}$）
   与极化塌缩判据（$\theta^*_{\Pi}$）给出的位置不同；模块度塌缩判据在小世界与实证双极
   网络上 **10/10 种子右删失**（残余社群结构造成结构性模块度下限），因此这三个判据
   回答的是不同问题。本报告同时给出三个判据及其删失比例，**不宣称存在唯一的
   「临界容忍度」**，论文中亦不再作「跨拓扑一致常数」的断言。

---

## 8. 复核后各假说的最终判定

判定口径与数据全部来自 `results/` 下的冻结产物，论文中的对应表格由
`scripts/generate_paper_tables.py` 自动生成，不存在人工转录。

<!-- AUTO: verdicts:BEGIN -->

### 8.1 预注册假说判定

| 假说 | 预注册判据 | 实测结果 | 判定 |
| :--- | :--- | :--- | :--- |
| **H1** 极化模块度跃迁 | $Q: 0.25 \to 0.65$（$T=20$ 内） | $Q_{\mathrm{end}} \in [0.174, 0.487]$，最大 $Q=0.500$；12 组配置通过率 $\leq 0%$ | **证伪 (Falsified)** |
| **H2** 桥接者脆弱性 | 桥接者边留存率至少低 $40\%$，$p<0.01$ | 桥接者 0.664 vs 其他节点 0.511；种子级 gap -0.299 (95% CI [-0.359, -0.238])，种子级 $p=1.000$ | **证伪 (Falsified)** |
| **H3** 反事实去极化 | 到达率 $\geq +35\%$ **且** $\Delta Q \leq -0.20$ | Policy 1：$\Delta Q=-0.038$（$p=7.4e-08$）、到达率 -12.3%（$p=0.72$）；Combined：$\Delta Q=-0.041$（$p=1.7e-06$）、到达率 +19.4%（$p=0.04$）。合取判据的两个前件均未达到 | **不成立 (Not supported)** |

### 8.2 RQ1 归因判定（Phase 4 受控归因消融）

受控因子设计：同一 seed 下所有实验臂共享同一初始图与同一初始信念，互惠边初始权重对称化。20 个种子 × 30 轮 × 60 节点。

| 实验臂 | $A_{\mathrm{rupture}}$（均值 $\pm$ 95% CI） | 恰为 0 的种子比例 | 说明 |
| :--- | :--- | :--- | :--- |
| `scalar_sync` | 0.000 $\pm$ 0.000 | 100% | 标量 HK + 同步调度：理论预期严格对称 |
| `isotropic_sync` | 0.000 $\pm$ 0.000 | 100% | 多维 $d{=}3$ + 同步 + **各向同性**权重（对照臂） |
| `multidim_sync` | 0.216 $\pm$ 0.051 | 0% | 多维 $d{=}3$ + 同步 + **个体各向异性**（仅比对照臂多一个变量） |
| `scalar_async` | 0.059 $\pm$ 0.021 | 15% | 标量 HK + 异步调度：隔离调度混杂项 |
| `multidim_async` | 0.228 $\pm$ 0.062 | 5% | 多维 + 异步 + 各向异性（原始量级） |
| `multidim_async_orig` | 0.134 $\pm$ 0.038 | 0% | 多维 + 异步 + 原始容忍度规则（参照） |

| 受控对比 | $\Delta A_{\mathrm{rupture}}$ | 配对 Wilcoxon $p$ | 正向种子数 |
| :--- | :--- | :--- | :--- |
| 调度效应（标量）：异步 − 同步 | +0.059 | 0.000291 | 17/20 |
| 各向异性效应（同步）：各向异性 − 各向同性 | +0.216 | 1.91e-06 | 20/20 |
| 调度效应（多维）：异步 − 同步 | +0.011 | 0.748 | 11/20 |
| 容忍度规则效应（异步）：原始规则 − 听者依赖 | -0.093 | 0.000851 | 5/20 |

**判定：`SUPPORTED_WITH_SCOPE`** —— 同步调度下，标量模型与各向同性多维模型的断裂严格对称（A≡0），仅当引入『个体各向异性语义判据』后才产生非对称撕裂；同时异步调度本身即可在标量模型中制造非对称断裂，故原论文把 A_rupture 全部归因于『多维语义』属于过度归因，正确表述为『个体各向异性语义判据』与『异步调度』的联合效应。

### 8.3 模块度上界一致性审计判定（Phase 5）

- 基准矩阵 **42** 个（数据集 × 方法）单元逐格核验 $Q \le 1 - 1/K$：**0 个违反**。
- 演化网格 **12** 组参数配置全部合规 （`all_grid_cells_within_bound = True`）。
- 引擎实测最大社群数 $K_{\max}=3$；没有任何配置达到 $Q>0.65$（`any_grid_config_reaching_q_above_0.65 = False`）。
- 因此 H1 的证伪理由是**经验性的、可核验的**，而不是「数学上不可能」。

### 8.4 跨拓扑临界容忍度判定（Phase 1c 临界点测量）

原正文断言「四个拓扑的临界容忍度一致落在 $\theta^*\in[0.50, 0.55]$」所依据的 $\theta$ 采样网格在该区间内只有一个点，**原理上无法判定该命题**；本轮以步长 0.05 的细网格（θ ∈ [0.15, 0.85]）重做测量，并给出三个各自有物理含义的判据，逐种子估计后跨种子汇总。

| 判据 | 跨拓扑 θ* 范围 | 拓扑间极差 | 落在预登记区间内的种子估计 | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| 连通性恢复阈值 θ*<sub>frag</sub>（巨型分量 ≥ 0.99） | [0.249, 0.700] | 0.186 | 1/40 (2%) | **与预登记区间不一致** |
| 模块度塌缩阈值 θ*<sub>Q</sub>（Q 跌破起点一半） | [0.453, 0.850] | 0.235 | 2/40 (5%) | **与预登记区间不一致** |
| 极化塌缩阈值 θ*<sub>Π</sub>（极化跌破起点一半） | [0.291, 0.582] | 0.163 | 3/40 (8%) | **与预登记区间不一致** |

| 拓扑 | θ*<sub>frag</sub> | θ*<sub>Q</sub> | θ*<sub>Π</sub> | θ*<sub>frag</sub> 删失种子 |
| :--- | :--- | :--- | :--- | :--- |
| `erdos_renyi` | 0.334 $\pm$ 0.034 | 0.620 $\pm$ 0.022 (0/10 删失) | 0.370 $\pm$ 0.037 | 0/10 |
| `scale_free` | 0.400 $\pm$ 0.081 | 0.615 $\pm$ 0.055 (0/10 删失) | 0.330 $\pm$ 0.024 | 0/10 |
| `small_world` | 0.357 $\pm$ 0.075 | 0.850 $\pm$ 0.000 (10/10 删失) | 0.493 $\pm$ 0.057 | 0/10 |
| `empirical_bipolar` | 0.520 $\pm$ 0.099 | 0.850 $\pm$ 0.000 (10/10 删失) | 0.373 $\pm$ 0.055 | 0/10 |

**判定**：原断言「θ* 一致落在 [0.50, 0.55]」**被证伪** —— 三个判据均无法支持它。真正稳定的相变是**连通性恢复**（碎裂→连通）：θ*<sub>frag</sub> 跨拓扑位于 [0.249, 0.700]，拓扑间极差仅 0.186；而模块度塌缩判据在小世界 / 实证双极网络上**大面积右删失**（Q 存在结构性下限，残余社群结构不随共识消失），说明用「Q 塌缩」定义临界点本身就不适用于这两类拓扑。RQ2 结论按实测值改写。

### 8.5 真实 LLM 认知层判定（Phase 6，同 seed 配对对照）

把旗舰引擎的认知后端从规则换成**真 LLM**（`MiniMax-M3.1-Flash`），两臂共享同一 seed（即同一初始图、同一初始信念、同一随机流），每轮一次批量调用，轮内命中即采用。统计单位为 seed（n = 10）。

| 指标 | 配对差（LLM − 规则） | 95% CI | $p$ | 显著？ |
| :--- | ---: | :--- | ---: | :--- |
| 模块度 $Q$ | -0.0040 | [-0.0107, +0.0107] | 0.418 | 否 |
| 极化 $\Pi$ | -0.0056 | [-0.0399, +0.0399] | 0.757 | 否 |
| 巨型分量占比 | +0.0000 | [-0.0056, +0.0056] | 1.000 | 否 |
| 撕裂非对称度 $A_{\mathrm{rupture}}$ | +0.0000 | [+0.0000, -0.0000] | 1.000 | 否（配对差全为 0，显式取 $p=1$） |
| 边数 | +1.1000 | [-14.1451, +14.1451] | 0.864 | 否 |
| 桥接者边留存率 | +0.0088 | [-0.0094, +0.0094] | 0.064 | 否 |

**覆盖率（必须显式披露）**：LLM 计划事件 1800，实际落地决策 1721，引擎互动总数 68404，**覆盖率 2.52%**；计划命中率 95.6%，解析成功率 98.3%（30 次解析失败显式计入并回退为 keep）。

**动作分布**：keep 1631 / mute 61 / unfollow 28 / block 1——模型确实会终止关系，而不只是重加权。

**收据**：实验共消耗 300 次 API 调用（= 300 个去重请求，每个请求恰好一次调用；验证重跑全部命中内容寻址归档，API 成本为 0，缓存命中 300 次）、token 计数 input 297691 / output 283085 / thinking 206722；响应按内容寻址归档，重跑为 0 次 API 调用、逐位可复现。

**判定**：**否**——在 2.52% 的互动覆盖率下，真 LLM 认知层**没有显著改变**结构 regime（模块度与撕裂非对称度的配对差均不显著）。这是一个**负结果**，照实报告：它否证了「换成真 LLM 就会改变拓扑结论」这一隐含预期，并把解释责任推回「覆盖率是否过低」与「效应量是否过小」这两个可检验问题。

### 8.6 网络规模敏感性判定（Phase 7，双臂设计）

引擎默认初始拓扑为固定 $p=0.08$ 的 Erdős–Rényi，因此 $|E|=O(N^2)$，**朴素扩规模会同时放大密度**。为把规模与密度解耦，Phase 7 v2 跑两个只差一个参数的臂：**密度匹配臂**（固定平均入度 $\bar d$，$p(N)=\bar d/(N-1)$，为主结果）与**固定-$p$ 臂**（$p=0.08$ 恒定，为混淆探针）。

| $N$ | 密度匹配 $Q$ | 固定-$p$ $Q$ | 固定-$p$ $\Pi$ | 固定-$p$ $A_{\mathrm{rupture}}$ | 固定-$p$ 桥接者 gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 0.427 | 0.444 | 0.261 | 0.089 | +0.121 |
| 100 | 0.446 | 0.446 | 0.163 | 0.121 | +0.146 |
| 250 | 0.455 | 0.273 | 0.000 | 0.367 | +0.115 |
| 500 | 0.456 | 0.141 | 0.000 | 0.681 | +0.033 |
| 1000 | 0.453 | 0.077 | 0.000 | 0.851 | -0.016 |

密度匹配臂的 $\log N$ 回归斜率 **+0.0082**（95% CI [+0.0006, +0.0159]，显著：是）；固定-$p$ 臂斜率 n/a（不显著）。

**真实 SNAP Higgs $N=1000$ 诱导子图**（10 种子）：$Q=0.489$、巨型分量占比 0.990，与密度匹配臂同量级。

**判定**：① 密度匹配下 $Q$ 在 $N=60\to1000$ 保持稳定，**「H1 的证伪只是小图假象」这一反驳不成立**；② 固定-$p$ 臂的 $Q$ 从 0.446 塌到 0.077、$\Pi\to0$、$A_{\mathrm{rupture}}\to0.85$、桥接者 gap 转负，但这是**纯密度伪影**（平均入度从 ≈5 涨到 ≈80），不是机制在大图上失效——因此论文**同时报告两臂**，并把「朴素扩规模会制造塌陷」本身作为一个方法学警示写进正文。

<!-- AUTO: verdicts:END -->

---

## 9. 复核产物清单

- 源码修复：`semantic.py`、`metrics.py`、`expkit.py`、`phase3.py`、`engine.py`、`graph.py`、`benchmark_suite.py`、`datasets_real.py`（新增）
- 新增实验模块：`phase4_attribution.py`、`phase5_bound_audit.py`、`phase1c_threshold.py`、`llm_backend.py`、`phase6_llm_engine.py`、`phase7_scaling.py`
- 新增工具：`scripts/generate_paper_tables.py`、`scripts/record_provenance.py`、`scripts/check_paper_integrity.py`（无 TeX 环境下校验宏 / label / `\input` / 图文件 / 自动生成标记 / 环境配对 / 数学模式污染）、`scripts/build_submission.py`（单文件投稿包拍平 + `--check`）、`scripts/check_page_budget.py`（ICWSM 11 页门禁 + 编译日志门禁）、`scripts/generate_figures.py`、`scripts/pdf_page_map.py`、`scripts/pdf_dump_pages.py`、`scripts/pdf_layout_probe.py`、`scripts/paper_wordcount.py`
- 新增测试：`tests/test_research_invariants.py`（9 项）
- 新增图表：`results/figures/fig7_attribution_ablation.{png,svg}`；`fig1` 重做为三面板（含跨拓扑相变面板）；`fig8_llm_arm_and_scaling.{png,svg}`（LLM 臂 + 规模/密度混淆）
- 自动生成表格：`paper/tables/sota_matrix.tex`、`sota_matrix_compact.tex`、`hypothesis_verdicts.tex`、`attribution.tex`、`dataset_metrics.tex`、`llm_arm.tex`、`scaling.tex`、`macros.tex`
- 投稿合规：`paper/checklist.tex`（ICWSM 官方 30 问逐条作答）、`paper/venue/aaai25.{sty,bst}`、`build/icwsm/NetworkEvolveLab_ICWSM.pdf`（正文+参考文献 11 页 / Checklist 11–13 页 / 附录 14–22 页）
- 溯源清单：`results/PROVENANCE.json`
