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
    parser.add_argument(
        "--reference-counterfactual-view",
        action="store_true",
        help="Fuse Stage2 FG prototypes with a background-suppressed reference view.",
    )
    parser.add_argument(
        "--reference-counterfactual-blend",
        default=0.5,
        type=float,
        help="Counterfactual contribution to Stage2 FG prototypes.",
    )
    parser.add_argument(
        "--hypergraph-tv",
        action="store_true",
        help="Apply post-Part4 training-free hypergraph total-variation refinement.",
    )
    parser.add_argument("--hypergraph-tv-lambda", default=0.05, type=float)
    parser.add_argument("--hypergraph-tv-iterations", default=50, type=int)
    parser.add_argument("--hypergraph-tv-local-similarity", default=0.5, type=float)
    parser.add_argument("--hypergraph-tv-anchor-ratio", default=0.1, type=float)
    parser.add_argument("--hypergraph-tv-fg-anchor-margin", default=0.2, type=float)
    parser.add_argument("--hypergraph-tv-bg-anchor-margin", default=0.2, type=float)
    parser.add_argument("--hypergraph-tv-min-fg-view-reliability", default=0.7, type=float)
    parser.add_argument("--hypergraph-tv-primal-step", default=0.02, type=float)
    parser.add_argument("--hypergraph-tv-dual-step", default=0.02, type=float)
    parser.add_argument("--hypergraph-tv-tolerance", default=1e-4, type=float)
    parser.add_argument(
        "--hypergraph-tv-second-order", action="store_true",
        help="Add signed second-order local stencil regularization to Hypergraph TV.",
    )
    parser.add_argument("--hypergraph-tv-second-order-lambda", default=0.01, type=float)
    parser.add_argument("--hypergraph-tv-second-order-rgb-quantile", default=0.75, type=float)
    parser.add_argument(
        "--hypergraph-tv-evidence-interval", action="store_true",
        help="Use Part2/Part3 evidence disagreement to relax Hypergraph-TV fidelity.",
    )
    parser.add_argument("--hypergraph-tv-evidence-interval-max-width", default=0.15, type=float)
    parser.add_argument("--hypergraph-tv-evidence-interval-scale", default=0.3, type=float)
    parser.add_argument("--hypergraph-tv-evidence-interval-epsilon", default=0.1, type=float)

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
