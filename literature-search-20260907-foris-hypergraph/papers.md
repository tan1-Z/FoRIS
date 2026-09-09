# FoRIS 稀疏超图整合：文献筛选

检索目的：验证“原始 FoRIS + training-free sparse hypergraph consolidation”的方法依据、最近工作风险与实验对照。检索日期：2026-09-07。

评分为本任务相关性下的 Insight / Completeness / Numeric Evidence（1–5），不代表录用概率。

| 工作 | 来源/年份 | 类型 | I/C/E | 标签 | 本方案关系 |
|---|---|---|---|---|---|
| [FoRIS](https://arxiv.org/html/2609.03384v1) | arXiv 2026 | method + diagnostic | 5/4/4 | A | 直接基线；FC 仅 cluster scalar reweighting，细长结构仍断裂 |
| [INSID3](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf) | CVPR 2026 Oral | pure method | 5/5/5 | A | 单冻结 DINOv3 training-free ICS 的最近基础工作 |
| [Learning with Hypergraphs](https://proceedings.neurips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html) | NeurIPS 2006 | theory + method | 5/5/4 | A | normalized hypergraph Laplacian 和 transductive propagation 基础 |
| [Interactive Image Segmentation Using Probabilistic Hypergraphs](https://www.sciencedirect.com/science/article/pii/S0031320309004440) | Pattern Recognition 2010 | pure method | 5/5/5 | A | superpixel、soft incidence 与 hypergraph label propagation 的直接图像分割先例 |
| [Contextual Hypergraph Modeling](https://openaccess.thecvf.com/content_iccv_2013/html/Li_Contextual_Hypergraph_Modeling_2013_ICCV_paper.html) | ICCV 2013 | pure method | 4/5/4 | B | 像素/区域上下文超边与背景分离 |
| [HGNN](https://ojs.aaai.org/index.php/AAAI/article/view/4235) | AAAI 2019 | pure method | 5/5/5 | B | 高阶多模态关系建模，但依赖训练 |
| [DHGNN](https://www.ijcai.org/proceedings/2019/366) | IJCAI 2019 | pure method | 4/5/4 | B | 动态构图与 vertex/hyperedge 两阶段卷积，但依赖训练 |
| [Hypergraph Propagation and Community Selection](https://proceedings.neurips.cc/paper_files/paper/2021/file/1da546f25222c1ee710cf7e2f7a3ff0c-Paper.pdf) | NeurIPS 2021 | pure method | 5/5/5 | A | inter-/intra-image hyperedge、按需稀疏传播、匹配歧义控制 |
| [Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a.html) | ICML 2022 | theory + method | 5/5/5 | A | 组内方差驱动 nonlinear diffusion 与全局收敛依据 |
| [Hypergraph Convolutional Networks for WSSS](https://arxiv.org/abs/2210.05564) | arXiv 2022 | pure method | 3/3/3 | B | superpixel spatial/kNN 超图用于分割；需要训练，来源状态较弱 |
| [ZLaP](https://openaccess.thecvf.com/content/CVPR2024/papers/Stojni_Label_Propagation_for_Zero-shot_Classification_with_Vision-Language_Models_CVPR_2024_paper.pdf) | CVPR 2024 | pure method | 4/5/5 | B | 非参数、稀疏化 label propagation 的视觉基础，但任务是分类 |
| [Training-Free Message Passing for Hypergraphs](https://arxiv.org/abs/2402.05569) | arXiv 2024 | pure method | 4/4/4 | Risk | “training-free hypergraph message passing” 已被明确提出，不能作为宽泛新颖点 |
| [Hypergraph Vision Transformers](https://openaccess.thecvf.com/content/CVPR2025/html/Fixelle_Hypergraph_Vision_Transformers_Images_are_More_than_Nodes_More_than_Edges_CVPR_2025_paper.html) | CVPR 2025 | pure method | 5/5/5 | A | 视觉 token 动态高阶结构和无聚类构图；需要训练 |
| [Reasoning Mamba](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Reasoning_Mamba_Hypergraph-Guided_Region_Relation_Calculating_for_Weakly_Supervised_Affordance_CVPR_2025_paper.pdf) | CVPR 2025 | pure method | 4/5/5 | Risk | 已使用 DINO feature + hypergraph 做区域关系推理，但任务/监督不同 |
| [MFHS](https://www.sciencedirect.com/science/article/pii/S0031320325013846) | Pattern Recognition 2026 | pure method | 4/4/4 | Risk | foundation model + hypergraph segmentation 已存在，但为 SAM2 半监督训练 |
| [Hyper-FSAD](https://arxiv.org/abs/2605.10628) | arXiv 2026 | method + theory | 4/4/4 | Risk | DINOv3 + training-free sparse hyper matching 已存在，虽属 anomaly detection |

## 最近工作边界

直接新颖性风险最高的是 Hyper-FSAD、Training-Free Message Passing、Reasoning Mamba 与 MFHS。它们分别覆盖 DINOv3 training-free sparse hyper matching、无训练超图消息传递、DINO+超图区域关系、foundation model+超图分割。因此本方案不能依赖这些宽泛组合命名。

可防守的差异化需要同时成立：training-free ICS、reference-conditioned FG/BG anchors、FoRIS evidence 初始化、uncertainty-only residual、组级 conflict/path gate，以及与普通 graph/clique expansion 的严格对照。

## 理论与实现警告

经典 Zhou-style 线性超图传播可能等价于某种 reweighted clique expansion。只使用 (HWD_e^{-1}H^T) 不足以证明真正高阶优势。需要 direct incidence 的 set-dependent gate 或 nonlinear hyperedge aggregation，并在相同节点、seed 和边预算下与 pairwise graph 比较。
