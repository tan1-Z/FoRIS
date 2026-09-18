# N2 + Hypergraph-TV Evidence Interval — Initial Configuration

This snapshot records the initial reproducible configuration for the symmetric
evidence-interval refinement.  The interval is enabled after Part4 and uses
the four existing maps `sf`, `1-sbn`, `cand_soft`, and `seed_prior` only to
measure disagreement.  It is not the later directional-interval experiment.

## Fixed model settings

| Setting | Value |
| --- | --- |
| DINO backbone | `large` |
| Image size | `1024` |
| COCO shots | `1` |
| Seed | `0` |
| Reference counterfactual view | enabled |
| Counterfactual blend | `0.25` |
| CRF refinement | enabled |
| Hypergraph-TV lambda | `0.05` |
| Hypergraph-TV iterations | `200` |
| Local similarity | `0.5` |
| Anchor ratio | `0.15` |
| FG / BG anchor margins | `0.20` / `0.20` |
| Minimum FG-view reliability | `0.70` |
| Primal / dual step | `0.02` / `0.02` |
| Tolerance | `0.0001` |
| Evidence interval max width | `0.15` |
| Evidence interval scale | `0.30` |
| Evidence interval epsilon | `0.10` |

## Initial COCO fold-0 command

```bash
CUDA_VISIBLE_DEVICES=3 python inference.py \
  --dataset coco \
  --data-root /home/lilinfei/INSID3/data \
  --fold 0 \
  --shots 1 \
  --seed 0 \
  --num-workers 0 \
  --model-size large \
  --device cuda \
  --crf-mask-refinement \
  --reference-counterfactual-view \
  --reference-counterfactual-blend 0.25 \
  --hypergraph-tv \
  --hypergraph-tv-lambda 0.05 \
  --hypergraph-tv-iterations 200 \
  --hypergraph-tv-local-similarity 0.5 \
  --hypergraph-tv-anchor-ratio 0.15 \
  --hypergraph-tv-fg-anchor-margin 0.20 \
  --hypergraph-tv-bg-anchor-margin 0.20 \
  --hypergraph-tv-min-fg-view-reliability 0.70 \
  --hypergraph-tv-primal-step 0.02 \
  --hypergraph-tv-dual-step 0.02 \
  --hypergraph-tv-tolerance 0.0001 \
  --hypergraph-tv-evidence-interval \
  --hypergraph-tv-evidence-interval-max-width 0.15 \
  --hypergraph-tv-evidence-interval-scale 0.3 \
  --hypergraph-tv-evidence-interval-epsilon 0.1 \
  --exp-name foris-n2-tv-evidence-interval-coco-fold0
```
