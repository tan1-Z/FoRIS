# Search notes

## Purpose

为原始 FoRIS 的 sparse hypergraph foreground consolidation 方案检索理论、视觉应用、training-free 传播、最近相似工作及新颖性风险。

## Public queries

- hypergraph label propagation image segmentation
- sparse hypergraph diffusion visual features
- DINO hypergraph segmentation
- training-free hypergraph message passing
- hypergraph few-shot / in-context segmentation
- clique expansion higher-order information loss

## Source policy

优先 NeurIPS、ICML/PMLR、CVF、AAAI、IJCAI、Pattern Recognition 和 arXiv 原文。聚合页只用于发现。排除 MDPI 与不可核验来源；题名去重。

## Main findings

1. 经典 normalized hypergraph propagation 可直接给出无训练二分类传播，但可能等价于 reweighted clique expansion。
2. ICML 2022 nonlinear hypergraph diffusion为 set-dependent group regularization 提供更强依据。
3. 视觉超图通常依赖训练；严格 training-free、reference-conditioned ICS 仍有可探索空间。
4. Hyper-FSAD 已覆盖 DINOv3 + training-free sparse hyper matching，宽泛组合不新。
5. 稀疏 incidence 只保证传播和存储为 (O(nnz(H)))；若 top-k 仍由全量相似度产生，构图计算仍可能接近 (O(N^2))。

## Opportunity map

- `crowded but open`：视觉超图和 foundation-model 超图已有大量方法，但多数需要训练。
- `mechanism gap`：FoRIS 的 fragmented foreground 与组级超边传播之间尚缺严格机制验证。
- `deployment/system gap`：dense clustering 与超图 sparse incidence 的真实端到端延迟/显存对比不足。
- `benchmark gap`：仅 mIoU 无法证明 topology restoration，需要 clDice、断裂和错误 bridge 指标。
- `negative-result opportunity`：若经典线性超图不优于普通图，可揭示 clique expansion 和 oversmoothing 局限，推动 nonlinear/path-gated 版本。

## Required novelty tests

- pairwise graph vs linear HG vs nonlinear HG；
- clique expansion vs direct incidence；
- 相同节点、seed、edge/membership budget；
- reference anchor/path edge 的必要性；
- 普通对象与细长结构分桶。
