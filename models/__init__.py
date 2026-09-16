"""Model construction utilities for INSID3."""

import torch

# from models.insid3 import INSID3
from models.foris import FoRIS

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": "/home/user9/dataset/user9/DINOV3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "/home/user9/dataset/user9/DINOV3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "/home/lilinfei/FoRIS/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _build_encoder(model_size: str = "large"):
    return torch.hub.load(
        "facebookresearch/dinov3",
        _HUB_NAMES[model_size],
        weights=_WEIGHTS[model_size],
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
    )
