# Search notes

## Purpose

为 FoRIS 的严格 training-free 多层、多尺度扩展确定：候选层、episode-wise selector、融合位置、分辨率策略和最近工作风险。

## Public queries

- DINOv3 dense features intermediate layers layer selection
- DINOv3 few-shot segmentation Semantic Selection Gap
- Hierarchical Layer Selection HERA DINOv3
- semantic correspondence DINO intermediate layer multi-layer feature fusion
- feature upsampling DINO dense prediction FeatUp ViT-Up
- multi-level correlation few-shot segmentation HSNet

## Screening policy

优先官方代码、CVF、ICLR、NeurIPS 与 arXiv 原文；搜索聚合页只用于发现，不作为最终主张证据。排除 MDPI 与无法核验来源。候选按题名去重。

## Main findings

1. FoRIS 的 layer sweep 证明细长结构最优层位于 19/20 附近，但最后层在普通任务上仍是稳定 anchor。
2. HERA 的 support leave-one-out ETR 是目前最贴近 mask 质量的 selector；其 TTA 部分不适用于严格 training-free FoRIS。
3. FSSDINO 表明 feature variance、entropy 等 heuristic 可能产生 selection regret，因此 selector 需要 deep prior、margin gate 和 soft fallback。
4. 对 FoRIS 最自然的融合位置是 affinity，其次是 robustly calibrated score；raw feature sum 仅应作基线。
5. learned upsampling 文献支持空间细节的重要性，但首版应使用 native-resolution ROI，以保留 single-backbone 主张。

## Opportunity map

- **Crowded but open**：episode-wise layer selection 已由 HERA 覆盖，但严格零更新且与 training-free ICS 三阶段原生耦合仍未被充分验证。
- **Mechanism gap**：已有 selector 很少分析为何某一层适合某类 geometry，以及 selector regret 如何按细长度、面积和域变化。
- **Deployment gap**：多层与 2048 输入的显存/延迟折中不足；ROI native-resolution 是可测量的系统贡献点。
- **Benchmark gap**：细长结构应补充 clDice、Boundary F1、连通性与 selection regret，而非仅报告 mIoU。

## Evidence cautions

- HERA 当前 arXiv 题名为 “Selective, Regularized, and Calibrated...”；其较早 OpenReview 投稿曾显示撤回，不应把该版本状态写成正式 ICLR 录用。
- ViT-Up 和 FSSDINO 是 2026 arXiv 工作，结论需标注为预印本证据。
- DPT、HSNet、FeatUp、ViT-Up、Diffusion Hyperfeatures 含训练组件，只能支持机制动机，不能证明 FoRIS 的 training-free 版本必然增益。
