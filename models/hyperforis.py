"""HyperFoRIS: training-free episode-adaptive hypergraph inference."""

from __future__ import annotations

from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hypergraph import build_relation_hyperedges, propagate_labels
from utils.data import build_transform, denormalize
from utils.refinement import crf_refine, init_crf, upsample_mask


@dataclass(frozen=True)
class HyperFoRISConfig:
    semantic_k: int = 8
    semantic_threshold: float = 0.15
    spatial_radius: int = 1
    cross_topk: int = 4
    background_modes: int = 8
    propagation_steps: int = 4
    propagation_alpha: float = 0.85
    background_score_weight: float = 1.0
    use_topology_hyperedges: bool = False
    adaptive_relation_weights: bool = False


class HyperFoRIS(nn.Module):
    """Frozen-DINOv3, reference/query hypergraph segmentation.

    Nodes are reference and query patch tokens. Hyperedges are rebuilt per
    episode and propagate foreground/background states without trainable heads.
    """

    def __init__(
        self,
        encoder: nn.Module,
        image_size: int = 1024,
        mask_refiner: str = "bilinear",
        resize_to_orig_size: bool = True,
        device: str = "cuda",
        **kwargs: object,
    ) -> None:
        super().__init__()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("HyperFoRIS was configured for CUDA, but CUDA is unavailable.")
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.image_size = image_size
        self.mask_refiner = mask_refiner
        self.resize_to_orig_size = resize_to_orig_size
        allowed = {field: kwargs[field] for field in HyperFoRISConfig.__dataclass_fields__ if field in kwargs}
        self.config = HyperFoRISConfig(**allowed)
        if mask_refiner == "crf":
            self._crf, self._crf_band_px, self._crf_p_core = init_crf(image_size, str(self.device))
        self._transform = build_transform(image_size)
        self._ref_images: torch.Tensor | None = None
        self._ref_masks: torch.Tensor | None = None
        self._tgt_image: torch.Tensor | None = None
        self._orig_tgt_size: tuple[int, int] | None = None
        self.last_hypergraph_info: dict[str, object] = {}

    def set_reference(self, image: str | Image.Image, mask: str | Image.Image | torch.Tensor) -> None:
        import numpy as np
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_tensor = self._transform(image).unsqueeze(0).to(self.device)
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.unsqueeze(0) if mask.ndim == 2 else mask
            mask_tensor = mask_tensor.to(self.device).float()
        else:
            if isinstance(mask, str):
                mask = Image.open(mask)
            mask_tensor = torch.from_numpy(np.array(mask.convert("L"), dtype="float32") / 255.0).unsqueeze(0).to(self.device)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0).clamp(0, 1)
        self._ref_images = image_tensor if self._ref_images is None else torch.cat((self._ref_images, image_tensor), dim=0)
        self._ref_masks = mask_tensor if self._ref_masks is None else torch.cat((self._ref_masks, mask_tensor), dim=0)

    def set_target(self, image: str | Image.Image, gt_mask: object | None = None) -> None:
        del gt_mask
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        self._orig_tgt_size = (image.height, image.width)
        self._tgt_image = self._transform(image).to(self.device)

    def segment(self) -> torch.Tensor:
        if self._ref_images is None or self._ref_masks is None or self._tgt_image is None:
            raise RuntimeError("Set at least one reference and one target before segment().")
        result = self.predict(self._ref_images, self._ref_masks, self._tgt_image)
        self._ref_images = self._ref_masks = self._tgt_image = None
        self._orig_tgt_size = None
        return result

    @torch.no_grad()
    def predict(self, ref_images: torch.Tensor, ref_masks: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        shots = ref_images.shape[0]
        target_batched = target_image.unsqueeze(0)
        images = torch.cat((ref_images, target_batched), dim=0).unsqueeze(0)
        features = self._extract_features(images)
        _, _, channels, height, width = features.shape
        ref_features = F.normalize(features[0, :shots], p=2, dim=1)
        query_features = F.normalize(features[0, shots], p=2, dim=0)
        # ref_masks is [S, H, W]; interpolate requires an explicit channel.
        occupancy = F.interpolate(
            ref_masks.float().unsqueeze(1), size=(height, width), mode="area",
        ).squeeze(1).clamp(0, 1)
        ref_nodes = ref_features.permute(0, 2, 3, 1).reshape(-1, channels)
        query_nodes = query_features.permute(1, 2, 0).reshape(-1, channels)
        ref_occupancy = occupancy.reshape(-1)

        edges, fg_seed, bg_seed, counts = build_relation_hyperedges(
            ref_nodes, query_nodes, ref_occupancy, (height, width),
            semantic_k=self.config.semantic_k,
            spatial_radius=self.config.spatial_radius,
            cross_topk=self.config.cross_topk,
            semantic_threshold=self.config.semantic_threshold,
            background_modes=self.config.background_modes,
            use_topology=self.config.use_topology_hyperedges,
        )
        num_ref, num_query = ref_nodes.shape[0], query_nodes.shape[0]
        initial = torch.zeros((num_ref + num_query, 2), device=self.device, dtype=query_nodes.dtype)
        initial[:num_ref, 0] = ref_occupancy
        initial[:num_ref, 1] = 1.0 - ref_occupancy
        initial[num_ref:, 0] = fg_seed
        initial[num_ref:, 1] = bg_seed
        is_reference = torch.zeros(num_ref + num_query, device=self.device, dtype=torch.bool)
        is_reference[:num_ref] = True
        relation_weights = self._relation_weights(edges.relation)
        weights = torch.stack(edges.weights) * relation_weights
        state = propagate_labels(
            initial, is_reference, edges.edges, weights,
            steps=self.config.propagation_steps, alpha=self.config.propagation_alpha,
        )
        score = (state[num_ref:, 0] - self.config.background_score_weight * state[num_ref:, 1]).view(height, width)
        mask = self._binarize(score, target_image.shape[-2:])
        result = self._finalize(mask, target_batched)
        self.last_hypergraph_info = {
            "method": "hyperforis", "nodes": int(num_ref + num_query),
            "reference_nodes": int(num_ref), "query_nodes": int(num_query),
            "relation_edges": counts, "total_hyperedges": len(edges.edges),
            "mean_hyperedge_size": float(sum(edge.numel() for edge in edges.edges) / max(1, len(edges.edges))),
            "relation_weights": {name: float(weight) for name, weight in self._relation_weight_map(edges.relation).items()},
            "fg_seed_fraction": float((fg_seed > 0).float().mean()),
            "bg_seed_fraction": float((bg_seed > 0).float().mean()),
        }
        return result

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        batch, count = images.shape[:2]
        flat = einops.rearrange(images, "b t c h w -> (b t) c h w")
        out = self.encoder.get_intermediate_layers(flat, n=1, reshape=True)[0]
        return einops.rearrange(out, "(b t) c h w -> b t c h w", b=batch, t=count)

    def _relation_weight_map(self, relations: list[str]) -> dict[str, torch.Tensor]:
        names = sorted(set(relations))
        if not names:
            return {}
        weights = {name: torch.tensor(1.0 / len(names), device=self.device) for name in names}
        if self.config.adaptive_relation_weights and "topology" in weights:
            weights["topology"] = weights["topology"] * 1.5
            total = sum(weights.values())
            weights = {name: value / total for name, value in weights.items()}
        return weights

    def _relation_weights(self, relations: list[str]) -> torch.Tensor:
        mapping = self._relation_weight_map(relations)
        return torch.stack([mapping[name] for name in relations])

    def _binarize(self, score: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        normalized = score - score.min()
        normalized = normalized / normalized.max().clamp_min(1e-6)
        up = F.interpolate(normalized[None, None], size=target_hw, mode="bilinear", align_corners=False)[0, 0]
        return up > 0.5

    def _finalize(self, mask: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        if self.mask_refiner == "crf":
            mask = crf_refine(self._crf, self._crf_band_px, self._crf_p_core, target_image, mask)
        if self.resize_to_orig_size and self._orig_tgt_size is not None:
            mask = upsample_mask(mask, *self._orig_tgt_size)
        return mask
