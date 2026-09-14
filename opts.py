"""Command-line arguments for INSID3 inference."""

import argparse

SUPPORTED_DATASETS = [
    "coco", "lvis", "pascal_part", "paco_part",
    "isaid", "isic", "lung", "suim", "permis", "fundus",
]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("INSID3 inference", add_help=False)

    # Model
    parser.add_argument(
        "--model-size",
        default="large",
        choices=["small", "base", "large"],
        help="DINOv3 backbone size",
    )
    parser.add_argument(
        "--image-size",
        default=1024,
        type=int,
        help="Input image resolution",
    )
    parser.add_argument(
        "--crf-mask-refinement",
        action="store_true",
        help="Enable CRF-based mask refinement.",
    )
    parser.add_argument("--dinov3-repo", default=None,
                        help="Local DINOv3 repository; defaults to DINOV3_REPO or GitHub.")
    parser.add_argument("--weights", default=None,
                        help="DINOv3 checkpoint; defaults to DINOV3_WEIGHTS or pretrain/.")
    # Adaptive Multi-Level + Local Token Refinement (all opt-in for baseline ablations).
    parser.add_argument("--adaptive_multilayer", action="store_true")
    parser.add_argument("--multilayer_ids", default=None,
                        help="Optional comma-separated zero-based DINO block ids.")
    parser.add_argument("--layer_fusion_temperature", default=0.5, type=float)
    parser.add_argument("--geometry_layer_weight", default=1.0, type=float)
    parser.add_argument("--use_soft_token_masks", action="store_true")
    parser.add_argument("--soft_fg_cluster_min", default=0.10, type=float)
    parser.add_argument("--local_refine", action="store_true")
    parser.add_argument("--local_refine_size", default=512, type=int)
    parser.add_argument("--local_refine_topk", default=3, type=int)
    parser.add_argument("--local_refine_quantile", default=0.90, type=float)
    parser.add_argument("--local_refine_margin_tokens", default=2, type=int)
    parser.add_argument("--local_refine_area_budget", default=0.25, type=float)
    parser.add_argument("--local_refine_alpha", default=0.5, type=float)
    parser.add_argument("--local_refine_miss_weight", default=0.5, type=float)
    parser.add_argument("--local_refine_boundary_weight", default=0.5, type=float)

    # Episode
    parser.add_argument(
        "--shots",
        default=1,
        type=int,
        help="Number of reference images (shots)",
    )

    # Hyperparameters
    parser.add_argument(
        "--svd-comps",
        default=500,
        type=int,
        help="Number of SVD components for positional debiasing",
    )
    parser.add_argument(
        "--tau",
        default=0.6,
        type=float,
        help="Clustering distance threshold",
    )
    parser.add_argument(
        "--merge-thresh",
        default=0.2,
        type=float,
        help="Cluster aggregation threshold",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="coco",
        choices=SUPPORTED_DATASETS,
        help="Dataset for evaluation",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="Fold index: for COCO and LVIS",
    )
    parser.add_argument(
        "--fundus-split",
        default="test",
        choices=["train", "test"],
        help="Fundus split to evaluate",
    )
    parser.add_argument(
        "--fundus-num-episodes",
        default=None,
        type=int,
        help="Number of sampled episodes for Fundus evaluation (default: real split size)",
    )

    # Runtime
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for logs and results",
    )
    parser.add_argument(
        "--exp-name",
        default="insid3-coco",
        help="Run name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of data loading workers",
    )

    return parser
