# FoRIS 的训练自由稀疏超图拓扑整合方案

> 工作名称：**SHyFoRIS — Sparse Hypergraph Foreground Consolidation for Training-Free In-Context Segmentation**。  
> 方案状态：研究与实施蓝图，尚无新增实验结果；所有数值阈值均为待验证初始值，不代表已取得性能提升。  
> 基线：FoRIS 官方仓库 `main`，固定 commit `1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e`。

## 1. 基线隔离与开发约束

已经建立两份相互独立的代码副本：

- `FoRIS-baseline-original`：从 [FoRIS 官方 GitHub](https://github.com/Xi-Mu-Yu/FoRIS) 克隆，固定在上述 commit，保持 clean，仅用于复现和对照；后续不修改该目录。
- `FoRIS-hypergraph`：从原始基线复制，开发分支为 `codex/hypergraph-consolidation`；后续超图代码只进入这一开发副本。

此前在主工作区进行的 CRGAF 多层修改不进入本方案。所有超图实验都从官方单层 FoRIS 开始，避免把选层、融合和超图三个变量混在一起。

建议立即给基线建立可核验记录：

```text
remote: https://github.com/Xi-Mu-Yu/FoRIS.git
branch: main
commit: 1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e
dirty: false
```

每次实验日志必须写入 baseline commit、开发 commit、数据 episode manifest、参数和环境版本。

## 2. 核心研究问题

### 2.1 问题

原始 FoRIS 的 FC（Foreground Consolidation）包括 Semantic Disagreement Penalization（SDP）与 Semantic Reweighting（SR）。SR 在 gated DINO 特征上再次做 agglomerative clustering，然后给每个 cluster 加一个统一标量。它有三个根本限制：

1. cluster 内部所有 patch 被同等修正，不能表达区域内部的方向、分支、端点和可信度差异；
2. cluster 之间没有显式高阶关系，无法恢复由多个局部片段共同组成的完整前景；
3. 层次聚类先构造 (N\times N) 距离矩阵，空间成本为 (O(N^2))，且 FC 与 FL 重复聚类。

### 2.2 核心假设

> FoRIS 的多个中间证据图已经提供“哪些 patch 可能属于前景”的语义线索；缺少的是一个能够把**语义相近、空间相关、共同受 reference 支持且满足结构连续性的一组 patch**作为整体传播的机制。稀疏超图可以用一个超边联合表示多个 patch，而无需把所有关系压成独立 pairwise edge。

### 2.3 最诚实的创新定位

最强贡献类型应限定为：

- **方法创新**：面向 training-free ICS 的 reference-conditioned sparse hypergraph consolidation；
- **经验/诊断创新**：说明何种高阶关系能恢复碎片化前景、何时会造成错误连接，以及其与普通图传播的差异。

不能只声称“首次把超图用于分割”。超图分割、超图神经网络和 DINO+超图均已有先例。真正需要证明的是：

1. 无参数更新；
2. reference-conditioned，而不是通用图像平滑；
3. 直接服务 FoRIS 的 fragmented foreground consolidation；
4. 使用稀疏 incidence 和不可被固定 pairwise graph 完全替代的组级可靠性机制。

## 3. 为什么使用超图，而不是普通稀疏图

普通图边只能表达两点关系：

```text
patch i ↔ patch j
```

但 FoRIS 中真正有用的证据往往是组关系：

```text
reference foreground prototype
    + 一组语义相似 target patches
    + 同一局部路径/区域中的邻居
    + 一致的候选投票
```

一个超边 (e\subseteq\mathcal V) 可以同时连接任意数量节点。例如，一条“道路片段”超边可包含两个高置信端点、间隔中的低置信 patch，以及对应的 reference prototype anchor。只有整组在语义、方向和局部外观上共同成立时，才允许跨缺口传播。

经典超图学习由 [Learning with Hypergraphs](https://proceedings.neurips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html) 给出规范化超图 Laplacian 与 transductive label propagation；[Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a.html)进一步指出，线性扩散难以直接利用节点特征，组内方差驱动的 nonlinear diffusion 可以提供更强的高阶约束。

一个关键风险是：经典线性算子

$$
\Theta=D_v^{-1/2}HWD_e^{-1}H^\top D_v^{-1/2}
$$

在某些条件下可解释为 reweighted clique expansion。若只计算 (HWH^\top) 后再调用普通图传播，方法的新颖性和高阶意义都很弱。因此本方案：

- 始终保存稀疏 incidence (H)，不物化 dense adjacency；
- 直接执行 vertex→hyperedge→vertex 两阶段传播；
- 最终版本让超边 gate 依赖整组节点的方差、标签冲突、路径方向和 anchor 支持，形成 set-dependent、迭代变化的传播；
- 必须与普通 kNN graph 和固定 clique expansion 做消融。

## 4. 与原始 FoRIS 的集成位置

原始流程保持：

```text
DINOv3 feature
  → FP: APD + foreground purification
  → FL: candidate voting + seed-cluster prior
  → S², sf, sb, V, P, gated target feature
  → FC: SDP + SR
  → mask
```

第一版只替换 FC 中的 SR：

```text
DINOv3 feature
  → 原始 FP（不改）
  → 原始 FL（不改）
  → 原始 SDP（不改）
  → Sparse Hypergraph Foreground Consolidation（替换 SR）
  → mask
```

这样可以把增益或退化明确归因于 hypergraph consolidation，而不是 backbone、选层、prototype 或 candidate voting 的变化。

后续再考虑 Full 版本：让超图同时替换 FL 的第二次 target clustering 和 FC 的 SR，从而真正消除两个 (N\times N) 聚类瓶颈。不能一开始就实现 Full 版本，否则实验无法回答“收益来自超图还是来自整个流程重写”。

## 5. 分阶段方法版本

### 5.1 HFC-Lite：替换 SR，验证方法有效性

- FP、FL、SDP 完全调用原始 FoRIS；
- target patch 是超图节点；
- 使用 `sf/sbn/V/P/S²` 产生正负 seed；
- 使用 gated target DINO feature、颜色和位置构建稀疏超边；
- 线性超图 label propagation 得到 (F_{hg})；
- 只在不确定 patch 上以 residual 方式修正原始 score。

该版本仍保留 FL 的原始 agglomerative clustering，因此主要验证精度和机制，不主张端到端复杂度已经降为 (O(Nk))。

### 5.2 HFC-Sparse：消除 FC 的 dense clustering

- 完全删除 `_semantic_cluster_reweight_map()` 的 agglomerative clustering；
- semantic hyperedge 用 chunked top-k 或 FAISS approximate kNN 构建；
- incidence 非零元素约为 (O(Nk))；
- 传播只使用 `scatter_add`，不依赖 PyG/DGL。

### 5.3 HFC-Topology：增加路径型超边

- 仅在 reference mask 显示细长/分支结构时开启；
- 加入方向、端点、允许短缺口的 path hyperedge；
- 使用组级 conflict gate 阻止错误 bridge；
- 主要目标是 Fundus、DeepGlobe-18 等细长结构，而不是强迫所有普通对象受拓扑约束。

### 5.4 Full-SHyFoRIS：统一 FL 与 FC

- 保留 FP 和 cross-image candidate voting；
- 用 reference-conditioned hypergraph 同时生成 seed prior 和 consolidated mask；
- 移除 FL/FC 的两次 agglomerative clustering；
- 只有前三个版本均通过后才实施。

## 6. 节点设计

### 6.1 Target patch 节点

每个 DINO patch 对应节点 (v_i)。在 1024 输入、patch size 16 时，约有 (64\times64=4096) 个节点。

节点属性：

$$
x_i=[\hat f_i;\ c_i;\ p_i;\ s_i^f;\ s_i^b;\ V_i;\ P_i;\ S_i^{(2)}],
$$

其中：

- (hat f_i)：原始 FoRIS FP gate 后的 DINO 特征；
- (c_i)：下采样 RGB/Lab 颜色；
- (p_i)：归一化二维坐标；
- (s_i^f,s_i^b)：前景、背景相似度；
- (V_i)：cross-image candidate vote；
- (P_i)：seed-cluster prior；
- (S_i^{(2)})：FL 输出分数。

构图时不同模态不应直接无标度拼接。DINO、颜色、坐标和 evidence 分别归一化，并用于不同超边族或距离项。

### 6.2 Reference anchor 节点

为了让超图真正保持 in-context 属性，而不是退化成通用后处理，加入少量 clamped anchor：

- foreground anchor：原始 FoRIS 的 reference foreground cluster prototypes；
- background anchor：hard-negative background prototype；
- 多 shot 时每个 reference 保留独立 anchor，再增加一个聚合 anchor。

anchor 数量通常远小于 target patch 数。其标签固定为：

$$
Y(a_{fg})=[0,1],\qquad Y(a_{bg})=[1,0].
$$

若第一版希望最小改动，可以先不显式增加 anchor 节点，只将 prototype affinity 写入超边权重；但正式版本更推荐显式 anchor，因为它能构成 reference-target 高阶关系。

## 7. 超边设计

超边集合定义为：

$$
\mathcal E=\mathcal E_{spa}\cup\mathcal E_{sem}
\cup\mathcal E_{reg}\cup\mathcal E_{ref}\cup\mathcal E_{path}.
$$

### 7.1 Spatial hyperedges：局部连续性

以 target patch 为中心建立 3×3 邻域超边；可选增加稀疏 5×5 多尺度邻域。成员权重：

$$
h(v_j,e_i^{spa})=
\exp\left(-\frac{\|p_i-p_j\|^2}{\sigma_p^2}
-\frac{\|c_i-c_j\|^2}{\sigma_c^2}\right).
$$

只对不确定节点或粗 mask 邻域建立大窗口超边；高置信背景远区使用小窗口或不构边，控制规模并减少背景扩散。

### 7.2 Semantic kNN hyperedges：非局部同类关系

每个不确定节点 (v_i) 从 gated DINO feature 中查找 top-(k_s) 近邻：

$$
e_i^{sem}=\{v_i\}\cup\operatorname{TopK}_{j}
\langle\hat f_i,\hat f_j\rangle.
$$

建议初始搜索 `k_s ∈ {6, 8, 12, 16}`。必须设置：

- 最低 semantic similarity；
- 每个节点最大 membership；
- 对跨远距离连接施加较低 base weight；
- 若超边同时包含强 foreground seed 与强 background seed，直接删除或降权。

Semantic hyperedge 可以连接同一类别的多个实例，但也是错误 bridge 的主要来源。

### 7.3 Region hyperedges：区域级一致性

HFC-Lite 可直接把 FoRIS FL 已产生的 target cluster 当作超边，因此无需额外 region extraction。HFC-Sparse/Full 版本则使用：

- 多尺度规则 cell（2×2、4×4 patch）；或
- SLIC superpixel 映射到 DINO patch；或
- sparse kNN connected components。

正式效率结果不能继续依赖原始 agglomerative cluster，否则仍有 (O(N^2)) 预处理。

### 7.4 Reference-conditioned hyperedges：跨图像语义锚定

对于每个 foreground prototype (mu_j^{fg})，选择满足 candidate vote 且相似度最高的一组 target 节点：

$$
e_j^{ref}=\{a_j^{fg}\}\cup
\operatorname{TopK}_{i}\left[
\langle\hat f_i,\mu_j^{fg}\rangle
+\lambda_v(V_i-0.5)
\right].
$$

background anchor 同理，但仅连接 hard-negative target patch。推荐使用 sparsemax 或阈值+top-k，使无关 target patch 权重严格为零。近期 [Hyper-FSAD](https://arxiv.org/abs/2605.10628) 已在 DINOv3 training-free anomaly detection 中使用 Sparse Hyper Matching，因此本方法不能把“DINOv3 + sparse hyper matching”本身作为新颖点；差异必须在 ICS、前景/背景双 anchor、空间拓扑超边和 label propagation 上成立。

### 7.5 Path hyperedges：细长结构和短缺口

该超边仅由 topology mode 启用。步骤为：

1. 从 reference mask 计算面积、骨架长度、平均宽度、分支点和各向异性，得到 `thinness score`；
2. 在 target 的高置信前景和候选响应上估计局部主方向；
3. 沿 8 或 16 个离散方向寻找长度为 (L\) 的路径候选；
4. 仅当两端为高置信前景、路径上 DINO/reference affinity 持续较高、颜色/边缘不出现强断点时建立超边；
5. 最多允许 (g\) 个连续低置信 patch，防止无约束跨越背景。

路径超边权重可定义为：

$$
w(e^{path})=
q_{end}\cdot q_{dir}\cdot q_{sem}\cdot q_{img}\cdot(1-q_{conflict}),
$$

其中分别表示端点置信度、方向一致性、reference 语义支持、图像路径一致性和强前景/背景冲突。

## 8. Seed 与初始标签

先保留 FoRIS SDP，定义：

$$
S_{base}=S^{(2)}-\Pi.
$$

将 (S_{base}) 做 median/MAD robust normalization 后得到初始前景概率 (q_i)。硬 seed 不能只依赖一个 map：

### 8.1 正 seed

满足：

- (q_i) 位于 episode 的高分位数；
- `sf`、`V`、`P` 至少两项支持前景；
- background score 不高；
- 与某个 foreground anchor 相似。

### 8.2 负 seed

满足任一：

- (q_i) 位于低分位数且 `sbn` 高；
- 匹配 hard-negative background anchor；
- 位于多个 evidence map 一致拒绝的区域。

### 8.3 Soft label 与 seed clamp

所有 target 节点均可初始化：

$$
Y_i=[1-q_i,q_i],
$$

但只对高置信正/负 seed 设置 clamp confidence (c_i\in[0,1])。这样即使某个 episode 没有可靠硬 seed，传播仍有 FoRIS unary 作为 restart，不会产生全零解。

## 9. 超边权重与高阶可靠性

每个超边的静态初始权重：

$$
w_e^{(0)}=\lambda_{type(e)}
\exp\left(-\frac{\operatorname{Var}_e(f)}{\sigma_f^2}
-\frac{\operatorname{Var}_e(c)}{\sigma_c^2}
\right)q_e^{ref}q_e^{spatial}.
$$

最终版本增加动态 group conflict gate：

$$
g_e^{(t)}=
\exp\left(-\frac{\operatorname{Spread}_e(F^{(t)})^2}{\sigma_y^2}\right)
\left(1-\operatorname{Conflict}_e^{seed}\right),
$$

$$
w_e^{(t)}=w_e^{(0)}g_e^{(t)}.
$$

其中 `Spread` 可取组内前景概率的 robust max-min 或 MAD；`Conflict` 衡量同一超边中是否同时出现强正、强负 seed。这一权重依赖整个节点集合，而不是独立 pairwise similarity，是超图相对普通图最需要验证的机制。

## 10. 稀疏传播算法

### 10.1 稀疏存储

不构造 dense (H\in\mathbb R^{N\times M})，只保存 COO：

```text
vertex_index[nnz]
hyperedge_index[nnz]
incidence_value[nnz]
hyperedge_weight[M]
```

若每个节点产生一个大小 (k) 的 semantic edge，则 `nnz≈Nk`。对于 (N=4096,k=8)，约 3.3 万 membership，远小于 1677 万 pairwise matrix entries。

### 10.2 线性传播基线

采用归一化 vertex→edge→vertex：

$$
Z_e^{(t)}=D_e^{-1}H^\top D_v^{-1/2}F^{(t)},
$$

$$
\tilde F^{(t+1)}=D_v^{-1/2}HWZ_e^{(t)},
$$

$$
F^{(t+1)}=\alpha\tilde F^{(t+1)}+(1-\alpha)Y.
$$

再加入 seed clamp：

$$
F_i^{(t+1)}=(1-c_i)F_i^{(t+1)}+c_iY_i.
$$

初始建议搜索 `α ∈ {0.6,0.75,0.85,0.9}`，迭代 `T ∈ {3,5,10,20}`，同时设置收敛阈值。不能默认传播越深越好，超图扩散更容易 oversmoothing。

### 10.3 组级 nonlinear propagation

正式版本在每轮先计算组级动态 gate (g_e^{(t)})，并可用 weighted median/trimmed mean 替代普通均值。该设计的目标是：

- 一致超边快速传播；
- 包含强冲突 seed 的超边自动关闭；
- 一个异常 patch 不应拖动整个路径或语义组；
- 长路径不会因为多轮平均而吞并背景。

实现上仍使用两次 `scatter_add`，仅多出每个超边的 group statistics，不需要训练神经网络。

## 11. 与 FoRIS 分数的融合

不要直接用超图输出替换全部预测。推荐 residual、uncertainty-gated 更新：

$$
U_i=1-2|q_i-0.5|,
$$

$$
\Delta_i^{hg}=
\operatorname{logit}(F_{i,fg}^{(T)})-
\operatorname{logit}(q_i),
$$

$$
S_i^{(3)}=S_{base,i}
+\lambda_{hg}U_i\operatorname{clip}(\Delta_i^{hg},-\delta,\delta).
$$

这样高置信 FoRIS patch 基本保持不变，超图主要修正边界、断裂和证据冲突区域。它比整图强平滑更容易避免普通对象掉点。

可选地，对 path hyperedge 允许在低置信端点附近提高 (U_i)，但必须保持强背景 seed clamp。

## 12. 稀疏构图的计算实现

### 12.1 Semantic neighbor search

分两阶段：

- 可行性阶段：chunked cosine top-k，不保存完整 (N\times N) 矩阵；空间 (O(Nk))，计算仍近似 (O(N^2C))；
- 效率阶段：FAISS GPU/HNSW approximate kNN，使搜索接近 (O(N\log N)) 或依赖索引近似复杂度。

论文中不能把 chunked exact top-k 宣称为严格 (O(Nk)) 总时间，只能声称传播与存储为 (O(Nk))。

### 12.2 原生 PyTorch message passing

不建议第一版引入 PyG/DGL，避免额外 CUDA 编译和版本风险。二分类传播只需要：

```python
edge_msg = scatter_add(incidence * vertex_msg[vertex_ids], edge_ids)
vertex_msg = scatter_add(
    incidence * edge_weight[edge_ids] * edge_msg[edge_ids],
    vertex_ids,
)
```

需要使用 degree normalization、`clamp_min(eps)` 和 FP32 accumulation；DINO 特征可保持 FP16/BF16。

## 13. 开发副本的文件设计

所有实现仅进入 `FoRIS-hypergraph`：

```text
FoRIS-hypergraph/
├── models/
│   ├── foris.py                    # 原始 FoRIS，尽量保持最小 diff
│   ├── foris_hypergraph.py         # FoRIS + HFC wrapper/subclass
│   └── hypergraph_consolidation.py # seed、超边、传播、residual
├── utils/
│   ├── hypergraph.py               # SparseIncidence 与 scatter propagation
│   ├── knn.py                      # chunked/FAISS top-k
│   └── topology.py                 # thinness、方向、path hyperedge
├── inference_hypergraph.py
├── opts_hypergraph.py
├── tests/
│   ├── test_sparse_incidence.py
│   ├── test_hypergraph_propagation.py
│   ├── test_seed_clamping.py
│   ├── test_no_false_bridge.py
│   └── test_baseline_equivalence.py
└── configs/
    ├── paper_exact.yaml
    ├── hfc_lite.yaml
    ├── hfc_sparse.yaml
    └── hfc_topology.yaml
```

推荐通过组合而非直接重写原始模型：

```python
score2, evidence, gated_feature = original_foris_until_fl(...)
penalty = original_sdp(evidence)
score_base = score2 - penalty
delta_hg, diagnostics = hypergraph_consolidator(...)
score3 = score_base + delta_hg
```

若必须改 `models/foris.py` 才能暴露中间量，应只增加 `return_intermediates`，并用测试保证关闭 HFC 时输出与官方 commit bitwise 或数值容差一致。

## 14. 建议配置项

```text
--consolidation {original_sr,graph_lp,hg_linear,hg_nonlinear,hg_topology}
--hg-semantic-k 8
--hg-semantic-min-sim 0.55
--hg-spatial-window 3
--hg-alpha 0.85
--hg-iterations 10
--hg-tolerance 1e-4
--hg-residual-weight 0.20
--hg-residual-cap 0.20
--hg-positive-quantile 0.90
--hg-negative-quantile 0.10
--hg-update-uncertain-only
--hg-enable-reference-anchors
--hg-enable-path-edges
--hg-knn-backend {chunked,faiss}
--save-hypergraph-diagnostics
```

这些只是统一实验接口。默认值必须在 validation episode 上确定，不能使用 test mIoU 调参。

## 15. 实施顺序

### Phase 0：锁定基线与 episode

1. 在原始基线副本上复现论文单层 FoRIS；
2. 保存每个 episode 的 target/reference 文件名、class、fold；
3. 固定 `episode_manifest.jsonl`；
4. 保存原始 S¹/S²/S³、sf/sbn/V/P 和最终 mask 的少量 golden outputs；
5. 建立 `test_baseline_equivalence.py`。

通过条件：开发副本关闭超图时，与基线在相同 episode 上逐像素一致或只存在明确记录的浮点误差。

### Phase 1：普通 sparse graph 控制组

先实现 patch kNN graph label propagation。这不是最终方法，而是必要对照：如果超图不能超过普通图，就不能声称高阶建模必要。

### Phase 2：HFC-Lite 线性超图

1. 实现 COO incidence；
2. spatial + semantic + region + reference hyperedges；
3. 实现线性传播和 seed clamp；
4. 以 residual 取代原始 SR；
5. 输出每种超边数量、平均大小、纯度、传播变化量。

### Phase 3：组级可靠性与 nonlinear diffusion

1. 增加 hyperedge variance/conflict gate；
2. 对比 mean、trimmed mean、max-min/nonlinear update；
3. 测试 oversmoothing 与迭代深度；
4. 证明性能差异不是单纯增加邻接边数量。

### Phase 4：Topology path hyperedge

1. reference thinness detector；
2. target tangent/path proposal；
3. 短缺口桥接与强背景否决；
4. 在 Fundus、DeepGlobe-18 上加入 clDice、中心线召回和错误连接分析。

### Phase 5：Full-SHyFoRIS 与效率

1. 用 sparse hypergraph 替代 FL 的 agglomerative target clustering；
2. 共享 FL/FC incidence 与 neighbor index；
3. FAISS/HNSW；
4. 完整精度—延迟—显存 Pareto。

## 16. 实验设计

### 16.1 主基准

- 原论文八个主基准：LVIS-92ᶦ、COCO-20ᶦ、ISIC、SUIM、iSAID、X-Ray、PASCAL-Part、PACO-Part；
- 细长结构压力测试：Fundus、DeepGlobe-18；
- 1-shot 和 5-shot；至少 3 个固定 episode seed 或一个公开固定 manifest。

### 16.2 Baselines

| Baseline | 目的 |
|---|---|
| 官方 FoRIS | 唯一主基线 |
| FoRIS w/o SR | 测量原始 SR 的真实贡献 |
| FoRIS + CRF only | 区分边界 refinement 与结构传播 |
| FoRIS + sparse pairwise graph LP | 判断超图是否优于普通图 |
| FoRIS + clique-expanded hypergraph | 判断 direct incidence/group gate 是否必要 |
| FoRIS + linear hypergraph LP | 判断非线性组级机制是否必要 |
| FoRIS + nonlinear HFC | 完整非拓扑版本 |
| FoRIS + topology HFC | 完整版本 |

### 16.3 核心消融

1. 超边族：`spa`、`sem`、`reg`、`ref`、`path` 单独和组合；
2. anchor：无 anchor、foreground、foreground+background；
3. seed：仅 S²、证据共识、证据共识+clamp；
4. propagation：graph、linear HG、nonlinear HG；
5. 更新范围：全图与 uncertain-only；
6. hyperedge size (k)、迭代数 (T)、restart (alpha)；
7. group gate：variance、conflict、path consistency；
8. 原始 SR 与 HFC residual 并用/替换；
9. chunked exact kNN 与 approximate kNN；
10. 普通对象与细长结构分别统计。

### 16.4 指标

主指标：mIoU、FB-IoU。  
结构指标：Boundary F1/IoU、clDice、centerline precision/recall、连通分量误差、断裂数、错误 bridge 数。  
效率指标：总延迟、构图延迟、传播延迟、峰值显存、`nnz(H)`、超边数和平均 cardinality。  
诊断指标：seed precision/recall、hyperedge purity、前景从 S²→S³ 的新增/删除 patch、按不确定度和形状分桶的增益。

### 16.5 需要展示的可视化

- FoRIS S²、原始 SR、HFC 输出与 GT；
- positive/negative seed；
- 不同类型超边，以颜色区分；
- 成功恢复的断裂路径和被 conflict gate 拒绝的错误桥接；
- 超边内部概率随迭代变化；
- graph LP 与 hypergraph LP 在同一 episode 上的差异。

## 17. 成功门槛与停止条件

### Gate A：基线等价

关闭 HFC 时必须复现官方输出。失败则停止算法实验，先修工程差异。

### Gate B：超图必要性

HFC 必须在相同节点特征和 seed 下优于 sparse pairwise graph；否则“超图”只是表示替换，应降级为普通稀疏图方案或重新设计真正组级 operator。

### Gate C：不是只改善细线、损害主任务

至少应保证主八基准总体不出现不可接受的系统性退化，同时在 fragmented/slender 分桶上体现稳定优势。具体容差在实验前注册，不能看到结果后修改。

### Gate D：结构改善真实存在

若 mIoU 上升但 clDice、错误 bridge 和断裂数不改善，则不能声称恢复 topology；最多只能声称区域平滑。

### Gate E：效率主张成立

只有 Full-SHyFoRIS 移除 dense clustering，并使用稀疏 neighbor search 后，才能宣称端到端复杂度改进。HFC-Lite 只能声称传播矩阵稀疏。

## 18. 主要风险与应对

| 风险 | 类型 | 应对 |
|---|---|---|
| 超图退化为 clique-expanded graph | 方法/新颖性 | direct incidence、set-dependent gate、graph/clique 消融 |
| 错误 bridge 把相似背景连入目标 | 方法 | 双端点、方向、reference affinity、强背景否决、短缺口限制 |
| 多轮传播 oversmoothing | 方法 | restart、seed clamp、uncertain-only、少迭代、动态 conflict gate |
| semantic kNN 仍需 (O(N^2)) 搜索 | 系统 | 分块只解决内存；正式版本使用 FAISS/HNSW |
| patch token 已混合前景/背景 | 表征 | 超图只能缓解不能消除；后续结合高分辨率 ROI/中层特征 |
| 超参数过多破坏 training-free 可信度 | 证据 | 少量共享参数、固定 validation、跨域不重调、敏感性曲线 |
| DINO+hypergraph 已有相关工作 | 新颖性 | 聚焦 ICS、reference-conditioned 双 anchor、拓扑传播和零参数 FC |
| 只在 Fundus 有效 | 定位 | 可转为 topology-sensitive ICS 专项贡献，但不夸大通用 SOTA |

## 19. 最近工作与定位边界

- [FoRIS](https://arxiv.org/html/2609.03384v1)：直接 baseline，提出 progressive foreground refinement，但 FC 仍是 cluster scalar reweighting。
- [Learning with Hypergraphs](https://proceedings.neurips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html)：经典 normalized hypergraph learning 和 transductive classification。
- [Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a.html)：ICML 2022，高阶 nonlinear diffusion 与组内 variance regularization。
- [Hypergraph Propagation and Community Selection](https://proceedings.neurips.cc/paper_files/paper/2021/file/1da546f25222c1ee710cf7e2f7a3ff0c-Paper.pdf)：NeurIPS 2021，使用 inter-/intra-image hyperedges 和按需传播处理局部匹配歧义。
- [Contextual Hypergraph Modeling](https://openaccess.thecvf.com/content_iccv_2013/html/Li_Contextual_Hypergraph_Modeling_2013_ICCV_paper.html)：ICCV 2013，以像素/区域超图建模图像上下文。
- [Hypergraph Vision Transformers](https://openaccess.thecvf.com/content/CVPR2025/html/Fixelle_Hypergraph_Vision_Transformers_Images_are_More_than_Nodes_More_than_Edges_CVPR_2025_paper.html)：CVPR 2025，说明视觉 token 高阶关系与高效动态构图的近期方向，但需要训练。
- [Reasoning Mamba](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Reasoning_Mamba_Hypergraph-Guided_Region_Relation_Calculating_for_Weakly_Supervised_Affordance_CVPR_2025_paper.pdf)：CVPR 2025，DINO feature + hypergraph 用于区域关系推理，但任务和训练范式不同。
- [Hyper-FSAD](https://arxiv.org/abs/2605.10628)：2026 预印本，DINOv3 + training-free sparse hyper matching，虽为 anomaly detection，但对宽泛新颖性构成直接风险。
- [Training-Free Message Passing for Learning on Hypergraphs](https://arxiv.org/abs/2402.05569)：training-free hypergraph message passing 已存在，不能把“无训练超图传播”本身作为新贡献。

基于当前检索，较可信的差异化表述是：

> We formulate foreground consolidation in training-free in-context segmentation as reference-conditioned sparse hypergraph diffusion, where semantic, spatial, prototype, and path-consistency relations jointly gate higher-order propagation over uncertain target regions.

该表述仍需实验与更完整 novelty audit 支撑，不能在没有 graph/clique baseline 的情况下直接使用。

## 20. 推荐的最小可行实现

首个可执行版本应严格控制范围：

1. 原始 FoRIS FP、FL、SDP 不改；
2. target patch nodes；
3. spatial 3×3、semantic top-8、FoRIS region cluster、FG/BG reference anchor 四类超边；
4. evidence-consensus seeds；
5. sparse linear propagation，`T=5/10`；
6. seed clamp + uncertain-only residual；
7. 对照原始 SR、pairwise graph LP 和 clique-expanded HG；
8. 先跑 COCO fold-0、PASCAL-Part、Fundus 的固定小规模 manifest；
9. 通过后再加入 nonlinear group gate 与 path hyperedge。

这个顺序能最快回答三个决定性问题：超图是否比原始 SR 好、是否比普通图好、是否真的恢复结构而不是仅平滑 mask。

## 21. 从“SR 后超图修补”转向“超图重构对应与聚类”

### 21.1 当前 HFC-Lite 的实测诊断

以下是已提供的 COCO fold-0、1000 episode 结果：

| 版本 | mIoU | 平均模型时间 | 结论 |
|---|---:|---:|---|
| 原始 `original_sr` | 58.28 | 未在此表重测 | 当前开发副本的公平基线 |
| 直接 `sparse_hg` | 56.69 | 1468.5 ms | 直接替换 SR 明显掉点 |
| 保守参数 `sparse_hg` | 57.53 | 2103.8 ms | 降低扩散强度后回升，但仍低于 SR |
| `sr_plus_sparse_hg` | 58.17 | 2861.7 ms | 接近 SR，但仍低 0.11 点且明显更慢 |

最后一行最重要：超图作为 SR 后的补丁没有产生稳定正增益，还额外引入约 2.86 秒推理时间。当前实现每个 episode 都构建 4096 条 spatial 超边和 4096 条 semantic 超边，同时**保留**：

1. FL 中的原始 dense candidate correspondence；
2. FL 中的 agglomerative target clustering；
3. SR 中的第二次 agglomerative clustering；
4. 新增的 sparse hypergraph kNN 与传播。

它属于“叠加关系模块”，而不是“重构聚类与对应”。因此下一阶段不应继续试图用 residual weight 弥补，而要让同一套 hypergraph 同时承担 candidate localization、region grouping 和 consolidation，删除重复的 dense 操作。

### 21.2 新版本名称与目标

工作名称：**RCH-FoRIS — Reference-Conditioned Hypergraph Routing for Training-Free ICS**。

目标不是泛化地“使用超图”，而是解决原始 FoRIS 的两个可测瓶颈：

```text
原始 FL：dense patch matching + agglomerative region cluster
原始 FC：第二次 agglomerative cluster + scalar SR
```

替换为：

```text
一次 reference-conditioned sparse hypergraph
  ├─ 稀疏近邻对应 → candidate votes
  ├─ hyperedge overlap → region grouping / seed prior
  └─ 两阶段传播 → localization + consolidation
```

核心主张必须限定为：在相同冻结 DINOv3 特征和相同 reference mask 条件下，RCH-FoRIS 以单个共享稀疏结构替代 FL/FC 的重复聚类和 dense correspondence，并通过 reference-conditioned 高阶关系恢复碎片化前景。

### 21.3 总体数据流

```text
reference image/mask + target image
        │
        ├─ 冻结 DINOv3（一次特征提取）
        │
        ├─ 原始 APD + FP（保留）
        │    └─ FG/BG prototypes、Score 1、gated target feature
        │
        ├─ 稀疏 mutual ANN correspondence（替代 dense 4D matching）
        │    └─ vote map V、reference confidence
        │
        ├─ 构建一次 RCH incidence H
        │    ├─ reference anchor hyperedges
        │    ├─ candidate-conditioned semantic hyperedges
        │    ├─ boundary-aware spatial hyperedges
        │    └─ component/region hyperedges
        │
        ├─ Localization propagation
        │    └─ P_hg 取代原始 FL seed-cluster prior P
        │
        ├─ 原始 SDP（保留）
        │
        └─ Consolidation propagation（复用 H，重估超边可靠性）
             └─ Score 3 与最终 mask
```

这一结构保证：hypergraph 不是原始流程后的额外第四套关系，而是 FL 与 FC 共享的唯一关系表示。

### 21.4 稀疏对应：替代 4D dense matching

原始 candidate voting 为每个 reference 构造完整 target×reference patch similarity。新版本使用 reference token index：

1. 对每个 reference，将 debiased patch feature 与下采样 mask 写入 index；
2. 对每个 target patch，仅检索 top-1 或 top-(k_c) reference neighbor；
3. 该近邻属于 reference foreground 时产生 vote；
4. 只有 mutual-nearest 或 margin 足够大的匹配才计入高置信 vote；
5. 多 shot 使用严格多数 (V(i)>1/2)，偶数 shot 的平票拒绝。

记 target token 为 (f_i^t)，第 (s) 个 reference 的 ANN 返回为 (j_s^*(i))：

$$
V_i=\frac1S\sum_{s=1}^{S}
\mathbb{1}\left[M_s\left(j_s^*(i)\right)=1\right].
$$

实现模式必须明确区分：

| 模式 | 计算 | 作用 |
|---|---|---|
| `chunked_exact` | 分块矩阵乘法 | 数值等价对照；只降低峰值显存，不宣称总时间降阶 |
| `faiss_flat` | GPU exact inner-product index | 工程对照；同样不是严格降阶 |
| `faiss_ivf` / `hnsw` | approximate nearest neighbor | 只有此模式才可主张近似 (O(N\log N)) 查询行为 |

必须报告 ANN recall@1 相对 `chunked_exact` 的差异，否则不能把精度变化归因于 hypergraph。

### 21.5 一次共享的超图构建

节点仍是 target patch，加上 reference FG/BG prototype anchor。与 HFC-Lite 的根本差异是：**不是所有 4096 个 patch 都发起全部超边**。

定义 active center set：

$$
\mathcal U=
\{i\mid V_i>0\}
\cup\{i\mid q_i\in[q_l,q_h]\}
\cup\{i\mid \max_j\langle\hat f_i,\mu_j^{fg}\rangle>\tau_a\}.
$$

其中 (q_i) 为 FP unary probability。强背景远区不作为 semantic/spatial hyperedge 中心；它们可保留为负 seed，但不主动向外传播。

#### A. Reference anchor hyperedges

每个 FG prototype anchor 连接 top-(k_a) 个、同时满足 candidate vote 的 target token。BG anchor 只连接 hard-negative token。

```text
anchor + top-k_a supported target nodes
```

这是 in-context 语义的主要来源，必须用 family-wise normalization 保证不被大量 local hyperedge 淹没。

#### B. Candidate-conditioned semantic hyperedges

只以 ℕU 中节点为中心，查询 mutual top-(k_s) semantic neighbor。成员还要满足：

```text
cosine(feature_i, feature_j) ≥ τ_sem
且不是强背景 seed
且至少一个节点具备 reference support
```

对普通对象初始搜索：`k_s ∈ {2,4,6}`；细长结构先使用较小 (k_s)，避免跨分支错误连接。

#### C. Boundary-aware spatial hyperedges

只在 ℕU 或其一跳邻域建立 3×3 空间超边。若两个 patch 的原图颜色梯度/边缘强度高，则降低 incident weight。这样空间超边不再覆盖整张背景，也不会简单跨越明显边界。

#### D. Hypergraph component region edges

这是替代 agglomerative clustering 的核心：先由 semantic/spatial/reference hyperedge 的高置信 membership 得到稀疏 bipartite incidence，再在 target nodes 上做 union-find/connected components。

每个 component 成为一个 region hyperedge：

$$
e_r=\{v_i\mid i\in\operatorname{CC}(H_{high})\}.
$$

component 合并条件必须同时满足：

```text
semantic affinity 足够高
空间相邻或共享 reference anchor
不包含强 FG/BG seed conflict
```

这样 region 的数量与大小由稀疏关系决定，而非 sklearn 的全量 average-linkage。

### 21.6 Family-wise normalized propagation

现有 HFC-Lite 将所有 hyperedge 放入同一传播算子，4096 条 spatial/semantic edge 会压过约 5 条 reference edge。新版本先按类型计算独立消息：

$$
\mathcal P_r(F)=D_{v,r}^{-1/2}H_rW_rD_{e,r}^{-1}H_r^\top D_{v,r}^{-1/2}F,
$$

再做 family-wise 融合：

$$
\mathcal P(F)=
\frac{
\lambda_{ref}\mathcal P_{ref}(F)+
\lambda_{reg}\mathcal P_{reg}(F)+
\lambda_{sem}\mathcal P_{sem}(F)+
\lambda_{spa}\mathcal P_{spa}(F)}
{\lambda_{ref}+\lambda_{reg}+\lambda_{sem}+\lambda_{spa}}.
$$

初始 validation 搜索范围：

```text
λ_ref ∈ {0.35, 0.45, 0.55}
λ_reg ∈ {0.20, 0.30, 0.40}
λ_sem ∈ {0.10, 0.20, 0.30}
λ_spa ∈ {0.05, 0.15, 0.25}
```

每次组合归一化为 1。不是按超边数量分配权重，而是按“关系来源的可信度”分配权重。

### 21.7 两阶段超图推理

#### Stage 1：Localization propagation

保留原始 Score 1：

$$
S_i^{(1)}=\mathrm{FP}(i).
$$

用 (S^{(1)})、candidate vote (V)、FG/BG anchors 形成初始 label (Y^{loc})。经过少量传播得到 (P^{hg})：

$$
F_{loc}^{t+1}=\alpha_{loc}\mathcal P(F_{loc}^{t})+(1-\alpha_{loc})Y^{loc},
$$

$$
P_i^{hg}=F_{loc,i}^{(T_{loc})}[fg].
$$

它直接取代原始 FL 的 agglomerative seed prior：

$$
S_i^{(2)}=S_i^{(1)}+
\lambda_V(V_i-0.5)+\lambda_P(P_i^{hg}-0.5).
$$

#### Stage 2：Consolidation propagation

保留 SDP，但不再做第二次 clustering。计算 conflict-aware hyperedge gate：

$$
g_e=
\exp\left(-\frac{\operatorname{Var}_{v\in e}(F_{loc,v})}{\sigma_y^2}\right)
\left(1-\operatorname{Conflict}_e^{seed}\right).
$$

只有组内前景概率一致、且不同时包含强正/强负 seed 的超边才继续传播。得到 (F_{con}) 后：

$$
S_i^{(3)}=S_i^{(2)}-\Pi_i+
\lambda_C U_i
\left[\operatorname{logit}(F_{con,i}^{fg})-
\operatorname{logit}(P_i^{hg})\right].
$$

所有 correction 都先乘 episode-wise score scale，再进入 FoRIS 原始分数空间，避免 HFC-Lite 中 logit/raw-score 不匹配的问题。

### 21.8 缓存与复用

每个 episode 中只允许以下一次性操作：

| 资源 | 构建时机 | 复用位置 |
|---|---|---|
| DINO target feature | encoder forward 一次 | FP、ANN、构图、两次传播 |
| reference feature/prototype | `set_reference` 或 episode 开始 | candidate vote、anchor hyperedge |
| ANN index | reference/target token 准备后一次 | candidate voting、semantic kNN |
| incidence H | FL 前一次 | localization、consolidation |
| component labels | H 的 connected components 一次 | region edge、诊断、最终区域统计 |
| positional basis | model 初始化或磁盘 cache | 所有 episode APD |

这条约束是避免重回当前 “FL clustering + SR clustering + hypergraph kNN” 三重构图的关键。

### 21.9 复杂度账本

令 target token 数为 (N)，reference token 数为 (N_s)，语义近邻数为 (k)，超图 incidence 非零数为 (L)，传播轮数为 (T)。

| 操作 | 原始 FoRIS | RCH-FoRIS 目标实现 |
|---|---|---|
| candidate correspondence | (O(SN N_s C)) 4D dense similarity | approximate ANN search；精确模式仅作对照 |
| FL region grouping | (O(N^2)) memory/time agglomerative | union-find over (O(Nk)) sparse incidence |
| FC reweight grouping | 第二次 (O(N^2)) clustering | 复用同一 H 与 component edge |
| propagation | 无 | (O(TL))，`scatter_add` 两次/轮 |
| feature extraction | 一次 | 一次 |

严格表述：只有在 `faiss_ivf/hnsw` + sparse components 模式下，才可以讨论端到端近似稀疏复杂度。`chunked_exact` 只降低 dense matching 的峰值显存，并不消除 (O(N^2)) 计算。

### 21.10 开发顺序

#### R0：测量与固定 episode

1. 保存固定 `episode_manifest.jsonl`；
2. 记录每个阶段的时间：encoder、APD/FP、ANN、H build、component、propagation、CRF；
3. 记录 (N,L)、各类超边 cardinality、active center 数、ANN recall@1；
4. 将 `original_sr` 固定为同 manifest 的基线。

#### R1：稀疏对应替换，其他保持原样

替换 4D candidate matching 为 `chunked_exact` top-1，保持原始 agglomerative FL/SR。目标是证明预测等价或在容差内，且降低峰值显存。

#### R2：稀疏 component 替换 FL clustering

保留原始 SR；只将 FL 的 agglomerative seed prior 改为 hypergraph component prior。对照：原始 FL、pairwise component、hypergraph component。

#### R3：共享 H 替换 SR clustering

移除 SR 的第二次 agglomerative，使用 component/anchor hyperedge 的 consolidation propagation。此时仍保留原始 FL 的 seed prior 作为安全 fallback。

#### R4：统一 FL+FC

仅当 R2/R3 都不低于相应基线时，才让 (P^{hg}) 全面取代原始 prior，形成完整 RCH-FoRIS。

### 21.11 最小实验矩阵

| 运行 | FL prior | FC consolidation | 关系结构 | 要回答的问题 |
|---|---|---|---|---|
| B0 | 原始 agglomerative | 原始 SR | 原始 | 官方基线 |
| B1 | 原始 | 原始 SR | chunked exact ANN | 稠密 4D 是否只是内存问题？ |
| B2 | sparse components | 原始 SR | pairwise graph | 稀疏 component 是否足够？ |
| B3 | sparse components | 原始 SR | hypergraph | 高阶 group 是否优于 pairwise？ |
| B4 | sparse components | shared-H propagation | hypergraph | 是否可删除第二次 clustering？ |
| B5 | hypergraph prior | shared-H propagation | full RCH | 是否可统一 FL/FC？ |

每组必须报告：mIoU、FB-IoU、Boundary F1、Fundus/DeepGlobe 的 clDice、总延迟、各阶段延迟、峰值显存、ANN recall、(L\)、超边类型 cardinality。

### 21.12 停止条件与风险

- 若 B2（pairwise components）已显著优于 B0，而 B3 不优于 B2，则问题是稀疏 component，不是超图；应收窄论文到 sparse graph efficiency，不再强调 hypergraph。
- 若 B3 在 Fundus/DeepGlobe 的 clDice 提升但常规 mIoU 不提升，可定位为 topology-sensitive ICS，而不是通用 FoRIS 替代。
- 若 B4 延迟下降但 mIoU 稳定，效率贡献成立；若 mIoU 明显下降，保留原始 SR，停止统一替换。
- 若 approximate ANN recall@1 低，先提升 index 参数；不能用 hypergraph 补偿错误 correspondence。
- 若 family-wise normalization 后 reference edge 仍无贡献，需要检查 prototype/anchor 构造，而不是继续扩大 semantic (k)。

### 21.13 建议的首个可执行配置

R1 的唯一目标是验证稀疏 candidate matching，不追求超图增益：

```text
feature layer: 官方默认最后层
candidate backend: chunked_exact
top-1 vote: strict majority
FL/FC: 原始 agglomerative + original SR
```

R2 的首个 hypergraph component 配置：

```text
active centers: V>0 或 unary uncertainty ≥ 0.5
semantic k: 4
mutual kNN: true
semantic min similarity: validation quantile / 0.65 起始
spatial window: 3，仅 active-center 邻域
family weights: ref=.45, region=.30, semantic=.15, spatial=.10
localization α=.50, T=3
```

先在 COCO fold-0 的固定 manifest 上完成 B0–B3；仅有 B3 明确超过 B2 后，再投入 Fundus、DeepGlobe-18 与 nonlinear/path hyperedge。

## 22. 实验回顾与后续方向

### 22.1 已获得的负结果

当前开发阶段已观察到：

| 变体 | 相对原始 SR 的现象 | 解释 |
|---|---|---|
| R2：FL sparse hypergraph component | 基本无增益、无明显掉点 | 在 COCO 常规对象上，FL region grouping 不是主要性能瓶颈 |
| R3：FL shared H 替代 FC SR clustering | 掉点 | 共享 FL/FC 关系不足以表达 FC 的区域语义纯度 |
| 独立 FC sparse hypergraph SR | 掉点 | 多组关系的高阶重叠没有自然优于 FC 的单一语义 cluster |
| SR 后 HFC residual | 接近 SR，但更慢 | 额外关系模块没有提供足以抵消构图开销的有效信息 |

这些结果不意味着超图没有价值；它们说明在标准 COCO mIoU 协议下，当前 hypergraph 设计主要在重述已有 DINO 相似度和 FoRIS evidence，缺少新的判别信息。继续扩大超边数量、迭代数或传播范围大概率只会增加平滑和时延。

### 22.2 方向 A：Gate-aware SR（最高优先级）

原始 FC 的输入是 gate 后特征 \hat f_i=g_i f_i，但在 FC clustering 前又做 L2 normalization。若 (g_i>0)，有：

$$
\operatorname{norm}(g_i f_i)=\operatorname{norm}(f_i),
$$

因此 gate 的幅值不会影响 FC 的聚类距离。一个更直接、低风险且仍 training-free 的改进是保留原始 FC clustering，但显式把 gate 当作区域统计的可信度：

$$
\bar s_{fg}^{(j)}=
\frac{\sum_{i\in\mathcal G_j}g_i s_{fg,i}}
{\sum_{i\in\mathcal G_j}g_i+\epsilon},
$$

$$
\bar s_{bg}^{(j)}=
\frac{\sum_{i\in\mathcal G_j}g_i s_{bg,i}}
{\sum_{i\in\mathcal G_j}g_i+\epsilon}.
$$

再沿用原始 SR 公式，并可把回写量乘以节点自身的 (g_i)。它不引入新构图，也不会改变 FL，直接检验“FP gate 是否应进入 FC”的问题。

必须比较：原始 SR、只加 gate-weighted cluster mean、只加 gate-weighted node writeback、二者同时使用。若该方向有效，说明问题在于 FoRIS 没有把已提取的 foreground purity 充分传递到 FC，而不是区域连通性不足。

### 22.3 方向 B：Reference-conditioned background mixture / subspace

当前 FoRIS 用全 reference background 中最接近前景的 top 20% token 构造一个 hard-negative prototype。单个背景中心很难覆盖墙面、阴影、邻近物体、纹理和同类干扰物等多模态背景。

可构造多个 hard-negative background prototypes：

$$
\{\mu_{bg}^{(1)},\ldots,\mu_{bg}^{(K_{bg})}\},
$$

并将背景 penalty 改为温和的 max/log-sum-exp 或 support-validated mixture。一个安全版本是：仅在当前 target patch 对多个 BG prototype 都高相似、且前景 prototype 相似度不稳定时增加惩罚。

需要避免背景 prototype 数量越多就过度压低前景，因此 (K_{bg}\in\{1,2,4}) 应用 support leave-one-out 或 reference augmentation 选择。该方向直接解决原论文动机中的“视觉相似背景干扰”。

### 22.4 方向 C：Support-calibrated evidence weighting

FoRIS 的 Score 2 使用固定权重：

$$
S^{(2)}=S^{(1)}+\lambda_V(V-0.5)+\lambda_P(P-0.5).
$$

实际实现中 λ 值是全数据集固定常数。不同 episode 的 vote、prior 与 prototype score 可靠性差异很大。

可用 support-only leave-one-out（多 shot）或 mask-preserving augmentation（1-shot）为 λ_V、λ_P、背景权重和 threshold 选择一个小型候选集合。关键原则是：选择整个 evidence 配置，而不是仅按不可靠 heuristic 选择 DINO layer。

候选配置必须很少，例如：

```text
(λV, λP) ∈ {(0.10,0.10), (0.20,0.25), (0.30,0.10), (0.10,0.30)}
```

在 support pseudo-query 上比较 IoU 后选用。若最优配置没有稳定超过默认值，应回退默认配置。这给 FoRIS 增加 episode 自适应性，同时不会新建大型关系结构。

### 22.5 方向 D：Conditional topology repair，只服务细长结构

R2/R3 在 COCO 无增益不应阻止拓扑研究；原论文自己表明 Fundus 和 DeepGlobe-18 的主要问题是断裂、patch mixing 和缺少连续性。

更合适的拓扑模块不是全图超图传播，而是只在以下条件同时成立时连接短缺口：

```text
两端为高置信前景
局部切线方向一致
reference prototype affinity 持续较高
路径中没有强 background evidence
缺口长度不超过阈值
```

可用 geodesic shortest path、oriented region growing 或 Hessian/vesselness 辅助。该方向应以 clDice、中心线召回、断裂数和错误 bridge 数为主指标，不能只看 COCO mIoU。

### 22.6 方向 E：特征职责分离，而非全局多层融合

此前全局多层融合容易改变 Score 1、candidate vote、prior 和 FC，变量过多。更可控的版本是：

```text
最后层：reference-query 语义匹配与 candidate vote
中深层：只用于局部边界/细长结构诊断
原始最后层分数：始终作为最终语义 anchor
```

只有当中深层与最后层在不确定区域一致、且 support pseudo-query 支持时，中深层才参与局部 correction。这样可利用论文附录中 layer 19/20 对细长结构的价值，不再让中层干扰常规对象的全局语义。

### 22.7 推荐执行顺序

1. 先完成 Gate-aware SR；它是最小改动、最贴合现有负结果的方向。
2. 若 Gate-aware SR 无效，做 background mixture / subspace；它直接作用于 FP 的主要失败模式。
3. 再做 support-calibrated evidence weighting；需要固定 episode manifest。
4. 将 topology repair 限定到 Fundus/DeepGlobe-18，并单独报告结构指标。
5. 暂停全图 FL/FC 超图重构；除非后续在细长结构上出现清楚的 topology 指标增益。
