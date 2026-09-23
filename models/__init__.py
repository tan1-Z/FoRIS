"""Model construction utilities for INSID3."""

import os
from pathlib import Path

import torch

# from models.insid3 import INSID3
from models.foris import FoRIS

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DINOV3_REPO = _PROJECT_ROOT / "third_party" / "dinov3"

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": str(_PROJECT_ROOT / "pretrain" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "base": str(_PROJECT_ROOT / "pretrain" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "large": str(_PROJECT_ROOT / "pretrain" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}


def _build_encoder(model_size: str = "large"):
    repo_dir = Path(
        os.environ.get("DINOV3_REPO_DIR", str(_DEFAULT_DINOV3_REPO))
    ).expanduser()
    if not (repo_dir / "hubconf.py").is_file():
        raise FileNotFoundError(
            "DINOv3 source is required locally so inference does not depend on "
            "a GitHub connection. Clone or copy the DINOv3 repository to "
            f"{repo_dir}, or set DINOV3_REPO_DIR to a directory containing hubconf.py."
        )
    weight_path = Path(_WEIGHTS[model_size]).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"DINOv3 {model_size} weights were not found: {weight_path}. "
            "Place the checkpoint in FoRIS/pretrain or update _WEIGHTS."
        )
    return torch.hub.load(
        str(repo_dir),
        _HUB_NAMES[model_size],
        source="local",
        weights=str(weight_path),
    )




# def build_insid3(
#     *,
#     model_size: str = "large",
#     image_size: int = 1024,
#     svd_components: int = 500,
#     tau: float = 0.6,
#     merge_threshold: float = 0.2,
#     mask_refiner: str = "bilinear",
#     resize_to_orig_size: bool = True,
#     device: str = "cuda",
# ):
#     encoder = _build_encoder(model_size)
#     model = INSID3(
#         encoder=encoder,
#         image_size=image_size,
#         svd_components=svd_components,
#         tau=tau,
#         merge_threshold=merge_threshold,
#         mask_refiner=mask_refiner,
#         resize_to_orig_size=resize_to_orig_size,
#         device=device,
#     )
#     for param in model.parameters():
#         param.requires_grad = False
#     return model



def build_foris(
    *,
    model_size: str = "large",
    image_size: int = 1024,
    svd_components: int = 500,
    tau: float = 0.6,
    mask_refiner: str = "bilinear",
    resize_to_orig_size: bool = True,
    device: str = "cuda",
    reference_counterfactual_view: bool = False,
    reference_counterfactual_blend: float = 0.5,
    reference_counterfactual_adaptive_ensemble: bool = False,
    reference_counterfactual_adaptive_strength: float = 0.25,
    reference_counterfactual_adaptive_max_blend: float = 0.50,
    reference_counterfactual_blur_kernel: int = 33,
    hypergraph_tv: bool = False,
    hypergraph_tv_lambda: float = 0.05,
    hypergraph_tv_iterations: int = 50,
    hypergraph_tv_local_similarity: float = 0.5,
    hypergraph_tv_anchor_ratio: float = 0.1,
    hypergraph_tv_fg_anchor_margin: float = 0.2,
    hypergraph_tv_bg_anchor_margin: float = 0.2,
    hypergraph_tv_min_fg_view_reliability: float = 0.7,
    hypergraph_tv_primal_step: float = 0.02,
    hypergraph_tv_dual_step: float = 0.02,
    hypergraph_tv_tolerance: float = 1e-4,
    hypergraph_tv_second_order: bool = False,
    hypergraph_tv_second_order_lambda: float = 0.01,
    hypergraph_tv_second_order_rgb_quantile: float = 0.75,
    hypergraph_tv_evidence_interval: bool = False,
    hypergraph_tv_evidence_interval_max_width: float = 0.15,
    hypergraph_tv_evidence_interval_scale: float = 0.3,
    hypergraph_tv_evidence_interval_epsilon: float = 0.1,
):
    encoder = _build_encoder(model_size)
    model = FoRIS(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        mask_refiner=mask_refiner,
        resize_to_orig_size=resize_to_orig_size,
        device=device,
        reference_counterfactual_view=reference_counterfactual_view,
        reference_counterfactual_blend=reference_counterfactual_blend,
        reference_counterfactual_adaptive_ensemble=reference_counterfactual_adaptive_ensemble,
        reference_counterfactual_adaptive_strength=reference_counterfactual_adaptive_strength,
        reference_counterfactual_adaptive_max_blend=reference_counterfactual_adaptive_max_blend,
        reference_counterfactual_blur_kernel=reference_counterfactual_blur_kernel,
        hypergraph_tv=hypergraph_tv,
        hypergraph_tv_lambda=hypergraph_tv_lambda,
        hypergraph_tv_iterations=hypergraph_tv_iterations,
        hypergraph_tv_local_similarity=hypergraph_tv_local_similarity,
        hypergraph_tv_anchor_ratio=hypergraph_tv_anchor_ratio,
        hypergraph_tv_fg_anchor_margin=hypergraph_tv_fg_anchor_margin,
        hypergraph_tv_bg_anchor_margin=hypergraph_tv_bg_anchor_margin,
        hypergraph_tv_min_fg_view_reliability=hypergraph_tv_min_fg_view_reliability,
        hypergraph_tv_primal_step=hypergraph_tv_primal_step,
        hypergraph_tv_dual_step=hypergraph_tv_dual_step,
        hypergraph_tv_tolerance=hypergraph_tv_tolerance,
        hypergraph_tv_second_order=hypergraph_tv_second_order,
        hypergraph_tv_second_order_lambda=hypergraph_tv_second_order_lambda,
        hypergraph_tv_second_order_rgb_quantile=hypergraph_tv_second_order_rgb_quantile,
        hypergraph_tv_evidence_interval=hypergraph_tv_evidence_interval,
        hypergraph_tv_evidence_interval_max_width=hypergraph_tv_evidence_interval_max_width,
        hypergraph_tv_evidence_interval_scale=hypergraph_tv_evidence_interval_scale,
        hypergraph_tv_evidence_interval_epsilon=hypergraph_tv_evidence_interval_epsilon,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_foris_from_args(args):
    return build_foris(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=int(args.svd_comps),
        tau=args.tau,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        resize_to_orig_size=False,
        device=args.device,
        reference_counterfactual_view=args.reference_counterfactual_view,
        reference_counterfactual_blend=args.reference_counterfactual_blend,
        reference_counterfactual_adaptive_ensemble=args.reference_counterfactual_adaptive_ensemble,
        reference_counterfactual_adaptive_strength=args.reference_counterfactual_adaptive_strength,
        reference_counterfactual_adaptive_max_blend=args.reference_counterfactual_adaptive_max_blend,
        reference_counterfactual_blur_kernel=args.reference_counterfactual_blur_kernel,
        hypergraph_tv=args.hypergraph_tv,
        hypergraph_tv_lambda=args.hypergraph_tv_lambda,
        hypergraph_tv_iterations=args.hypergraph_tv_iterations,
        hypergraph_tv_local_similarity=args.hypergraph_tv_local_similarity,
        hypergraph_tv_anchor_ratio=args.hypergraph_tv_anchor_ratio,
        hypergraph_tv_fg_anchor_margin=args.hypergraph_tv_fg_anchor_margin,
        hypergraph_tv_bg_anchor_margin=args.hypergraph_tv_bg_anchor_margin,
        hypergraph_tv_min_fg_view_reliability=args.hypergraph_tv_min_fg_view_reliability,
        hypergraph_tv_primal_step=args.hypergraph_tv_primal_step,
        hypergraph_tv_dual_step=args.hypergraph_tv_dual_step,
        hypergraph_tv_tolerance=args.hypergraph_tv_tolerance,
        hypergraph_tv_second_order=args.hypergraph_tv_second_order,
        hypergraph_tv_second_order_lambda=args.hypergraph_tv_second_order_lambda,
        hypergraph_tv_second_order_rgb_quantile=args.hypergraph_tv_second_order_rgb_quantile,
        hypergraph_tv_evidence_interval=args.hypergraph_tv_evidence_interval,
        hypergraph_tv_evidence_interval_max_width=args.hypergraph_tv_evidence_interval_max_width,
        hypergraph_tv_evidence_interval_scale=args.hypergraph_tv_evidence_interval_scale,
        hypergraph_tv_evidence_interval_epsilon=args.hypergraph_tv_evidence_interval_epsilon,
    )
