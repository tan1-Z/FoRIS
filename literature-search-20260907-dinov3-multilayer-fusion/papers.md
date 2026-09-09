# DINOv3 多层选择与融合：文献筛选

用途：为 `FoRIS_创新点_局限与改进方向.md` 的 6.2 节提供可追溯依据。检索日期：2026-09-07。

评分为本次任务相关性下的 1–5 分：Insight / Completeness / Numeric Evidence。它评价论文对本方案的参考价值，不代表录用概率或研究质量的绝对排序。

| 论文 | 来源状态 | 类型 | I/C/E | 标签 | 与 FoRIS 6.2 的关系 |
|---|---|---|---|---|---|
| [FoRIS](https://arxiv.org/html/2609.03384v1) | arXiv 2026 v1 | method + analysis | 5/4/4 | A | 直接提供 DINOv3-L 的 layer/resolution sweep 和细长结构失效证据 |
| [DINOv3](https://arxiv.org/abs/2508.10104) | Meta technical report, 2025 | pure method | 5/5/5 | A | backbone、dense feature、Gram anchoring及高分辨率依据 |
| [DINOv3 official layer sets](https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/__init__.py) | 官方代码 | system/tool | 4/5/4 | A | ViT-L 的 LAST、FOUR_LAST、`[4,11,17,23]` 均匀抽层基线 |
| [HERA](https://arxiv.org/html/2605.19340) | arXiv 2026；早期 OpenReview 版本曾撤回，当前题名/版本以 arXiv 为准 | pure method | 4/5/4 | Risk | ETR、support leave-one-out、12–23 候选层和局部融合是最近的直接先验工作 |
| [Semantic Selection Gap / FSSDINO](https://arxiv.org/abs/2602.07550) | arXiv 2026 | method + diagnostic | 4/4/4 | Risk | 证明 oracle 中间层上限高，但启发式选择可能低于最后层 |
| [INSID3](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf) | CVPR 2026 Oral | pure method | 5/5/5 | A | FoRIS 的同 backbone training-free 基线与位置去偏来源 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | arXiv 2023/2024；官方 Meta 工作 | pure method | 5/5/5 | B | 通用 frozen dense feature 的基础证据 |
| [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814) | arXiv 2021/2022 | method + analysis | 5/4/4 | A | 层深、feature facet 与 positional bias 对 dense correspondence 的早期系统分析 |
| [A Tale of Two Features](https://arxiv.org/abs/2305.15347) | arXiv 2023 | pure method | 4/4/5 | A | 独立归一化后拼接互补 descriptor；DINO 的稀疏准确与空间连贯性权衡 |
| [Hypercorrelation Squeeze](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Hypercorrelation_Squeeze_for_Few-Shot_Segmentation_ICCV_2021_paper.html) | ICCV 2021 | pure method | 5/5/5 | A | 多层 correlation 而非 raw feature 混合的强先例，但依赖训练 decoder |
| [Vision Transformers for Dense Prediction](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html) | ICCV 2021 | pure method | 5/5/5 | B | 从多层 token 重组并逐级融合 dense prediction，依赖训练 decoder |
| [FeatUp](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html) | ICLR 2024 | pure method | 5/5/5 | A | 恢复低分辨率 feature 的空间细节，但会引入 learned upsampler |
| [ViT-Up](https://arxiv.org/abs/2606.14024) | arXiv 2026 | pure method | 4/4/4 | B | 用 ViT 中间 hidden states 做忠实 feature upsampling；是 ROI/native-resolution 的重要对照 |
| [Diffusion Hyperfeatures](https://proceedings.neurips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html) | NeurIPS 2023 | pure method | 5/5/5 | B | 证明跨层/尺度 feature aggregation 可搜索互补信息，但聚合器需要训练 |
| [Emergent Correspondence from Image Diffusion](https://proceedings.neurips.cc/paper_files/paper/2023/file/0503f5dce343a1d06d16ba103dd52db1-Paper-Conference.pdf) | NeurIPS 2023 | method + analysis | 4/5/5 | B | correspondence 的层/时间选择参考；模型和特征体系与 FoRIS 不同 |

## 最近工作风险

HERA 与 FSSDINO 是必须正面讨论的最近工作。HERA 已覆盖 per-episode support leave-one-out layer selection 和 local fusion；FSSDINO 已指出 oracle 中间层与可用 heuristic 之间的 Semantic Selection Gap。因此，FoRIS 扩展不能仅声称“首次自适应选中间层”。可区分的核心应是严格零参数更新、affinity-level 与 FP/FL/FC 的一体化融合、面向细长结构的 geometry-aware conservative routing，以及局部原生高分辨率策略。

## 证据边界

DPT、HSNet、Diffusion Hyperfeatures、FeatUp、ViT-Up 的结果支持多层或高分辨率特征有价值，但它们含可学习模块，不能直接证明 training-free FoRIS 一定提高。DINOv3 官方 `[4,11,17,23]` 组合也服务于训练后的 dense head，只应作为 baseline。最终层集和融合权重必须在 FoRIS 协议下重新验证。
