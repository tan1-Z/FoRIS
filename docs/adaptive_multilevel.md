# Adaptive Multi-Level + Local Token Refinement

This opt-in FoRIS extension stays training-free and keeps DINOv3 frozen.  It
uses support masks only: per-layer support separability, compactness, boundary
discriminability, and multi-shot consistency produce semantic and structural
qualities.  FP fuses per-layer *scores*; FL runs its quadratic correspondence
on one routed layer, and FC clusters one structural layer.

`--use_soft_token_masks` keeps fractional feature-token foreground occupancy,
which is particularly useful for thin structures.  `--local_refine` proposes at
most `--local_refine_topk` uncertain connected ROIs, batches 512px DINO crops,
and blends their prototype score back in pixel space with a Hann window.

Baseline:

```bash
python inference.py --model-size large
```

Full adaptive version:

```bash
python inference.py --model-size large --adaptive_multilayer --use_soft_token_masks --local_refine
```

All three flags are off by default; that retains FoRIS's original single last
layer feature extraction, FP, FL, FC, threshold, and finalization path.
