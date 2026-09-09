"""Model construction utilities for FoRIS."""

import os
from pathlib import Path
import torch

# from models.insid3 import INSID3
from models.foris import FoRIS

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHT_FILENAMES = {
    "small": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _build_encoder(model_size: str = "large", *, dinov3_repo: str | None = None,
                   weights: str | None = None):
    # return torch.hub.load(
    #     "/home/user9/dataset/user9/DINOV3/dinov3",
    #     _HUB_NAMES[model_size],
    #     weights=_WEIGHTS[model_size],
    # )
    repo = dinov3_repo or os.environ.get("DINOV3_REPO", "facebookresearch/dinov3")
    source = "local" if Path(repo).expanduser().exists() else "github"
    weights = weights or os.environ.get("DINOV3_WEIGHTS")
    if weights is None:
        local = Path("pretrain") / _WEIGHT_FILENAMES[model_size]
        weights = str(local) if local.is_file() else None
    if weights is None:
        raise FileNotFoundError("Pass --weights, set DINOV3_WEIGHTS, or place the checkpoint under pretrain/.")
    return torch.hub.load(repo, _HUB_NAMES[model_size], source=source,
                          trust_repo=True, weights=weights)




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
    adaptive_multilayer: bool = False,
    multilayer_ids: list[int] | None = None,
    layer_fusion_temperature: float = 0.5,
    geometry_layer_weight: float = 1.0,
    use_soft_token_masks: bool = False,
    soft_fg_cluster_min: float = 0.10,
    local_refine: bool = False,
    local_refine_size: int = 512,
    local_refine_topk: int = 3,
    local_refine_quantile: float = 0.90,
    local_refine_margin_tokens: int = 2,
    local_refine_area_budget: float = 0.25,
    local_refine_alpha: float = 0.5,
    local_refine_miss_weight: float = 0.5,
    local_refine_boundary_weight: float = 0.5,
    dinov3_repo: str | None = None,
    weights: str | None = None,
):
    encoder = _build_encoder(model_size, dinov3_repo=dinov3_repo, weights=weights)
    model = FoRIS(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        mask_refiner=mask_refiner,
        resize_to_orig_size=resize_to_orig_size,
        device=device,
        adaptive_multilayer=adaptive_multilayer,
        multilayer_ids=multilayer_ids,
        layer_fusion_temperature=layer_fusion_temperature,
        geometry_layer_weight=geometry_layer_weight,
        use_soft_token_masks=use_soft_token_masks,
        soft_fg_cluster_min=soft_fg_cluster_min,
        local_refine=local_refine,
        local_refine_size=local_refine_size,
        local_refine_topk=local_refine_topk,
        local_refine_quantile=local_refine_quantile,
        local_refine_margin_tokens=local_refine_margin_tokens,
        local_refine_area_budget=local_refine_area_budget,
        local_refine_alpha=local_refine_alpha,
        local_refine_miss_weight=local_refine_miss_weight,
        local_refine_boundary_weight=local_refine_boundary_weight,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_foris_from_args(args):
    ids = None if not args.multilayer_ids else [int(x.strip()) for x in args.multilayer_ids.split(",") if x.strip()]
    return build_foris(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=int(args.svd_comps),
        tau=args.tau,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        resize_to_orig_size=False,
        device=args.device,
        adaptive_multilayer=args.adaptive_multilayer,
        multilayer_ids=ids,
        layer_fusion_temperature=args.layer_fusion_temperature,
        geometry_layer_weight=args.geometry_layer_weight,
        use_soft_token_masks=args.use_soft_token_masks,
        soft_fg_cluster_min=args.soft_fg_cluster_min,
        local_refine=args.local_refine,
        local_refine_size=args.local_refine_size,
        local_refine_topk=args.local_refine_topk,
        local_refine_quantile=args.local_refine_quantile,
        local_refine_margin_tokens=args.local_refine_margin_tokens,
        local_refine_area_budget=args.local_refine_area_budget,
        local_refine_alpha=args.local_refine_alpha,
        local_refine_miss_weight=args.local_refine_miss_weight,
        local_refine_boundary_weight=args.local_refine_boundary_weight,
        dinov3_repo=args.dinov3_repo,
        weights=args.weights,
    )
