



from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import build_transform, denormalize, downsample_mask
from utils.refinement import crf_refine, init_crf, upsample_mask


class FoRIS(nn.Module):
    """Training-free in-context segmentation using a frozen DINOv3 encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        image_size: int = 1024,
        svd_components: int = 500,
        tau: float = 0.6,
        mask_refiner: str = "bilinear",
        resize_to_orig_size: bool = True,
        device: str = "cuda",
        cluster_logsumexp_temp: float = 0.07,
        dino_bg_weight: float = 0.55,
        use_raw_target_clustering: bool = True,
        use_raw_target_scoring: bool = True,
        candidate_boost: float = 0.20,
        seed_cluster_boost: float = 0.25,
        semantic_disagreement_weight: float = 0.08,
        semantic_bg_coupling_weight: float = 0.05,
        semantic_cluster_fg_boost: float = 0.20,
        semantic_cluster_conflict_suppress: float = 0.18,
        semantic_penalty_uncertainty_power: float = 1.5,
        semantic_penalty_max: float = 0.22,
        semantic_cluster_neg_cap: float = 0.12,
    ):
        super().__init__()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "INSID3 was configured for CUDA, but no CUDA device is available."
            )
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.image_size = image_size
        self.svd_components = svd_components
        self.tau = tau

        self.mask_refiner = mask_refiner
        self.resize_to_orig_size = resize_to_orig_size

        # Part 1: positional debiasing
        self.positional_basis = self._build_positional_basis(device)

        # Part 2: two-stage background suppression

        self.dino_bg_weight = float(dino_bg_weight)

        # Part 3: clustering
        self.cluster_logsumexp_temp = float(cluster_logsumexp_temp)
        self.enable_clustering = True
        self.use_raw_target_clustering = bool(use_raw_target_clustering)
        self.use_raw_target_scoring = bool(use_raw_target_scoring)

        self.candidate_boost = float(candidate_boost)

        self.seed_cluster_boost = float(seed_cluster_boost)

        # Part 4: semantic consistency correction
        self.semantic_disagreement_weight = float(semantic_disagreement_weight)
        self.semantic_bg_coupling_weight = float(semantic_bg_coupling_weight)

        self.semantic_cluster_fg_boost = float(semantic_cluster_fg_boost)
        self.semantic_cluster_conflict_suppress = float(semantic_cluster_conflict_suppress)
        self.semantic_penalty_uncertainty_power = float(semantic_penalty_uncertainty_power)
        self.semantic_penalty_max = float(semantic_penalty_max)
        self.semantic_cluster_neg_cap = float(semantic_cluster_neg_cap)


        if mask_refiner == "crf":
            self._crf, self._crf_band_px, self._crf_p_core = init_crf(
                image_size, str(self.device)
            )

        self._transform = build_transform(image_size)
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None

        self.should_debiass = True
        self.last_hg_part4_analysis = None
        self.last_hg_a1_analysis = None
        self.last_hg_a1_baseline_mask = None
        self.last_p1_diffusion_analysis = None
        self.last_p1_before_mask = None
        self.last_p1_2_mask = None
        self.last_p1_patch_attribution_state = None

    # ──────────────────────── Public API ────────────────────────

    def set_reference(
        self,
        image: str | Image.Image,
        mask: str | Image.Image | torch.Tensor,
    ) -> None:
        """Set reference image and mask from file paths or PIL Images."""
        import numpy as np

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_tensor = self._transform(image).unsqueeze(0).to(self.device)

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.unsqueeze(0) if mask.ndim == 2 else mask
            mask_tensor = mask_tensor.to(self.device)
            if mask_tensor.dtype != torch.bool:
                mask_tensor = mask_tensor > 0
        else:
            if isinstance(mask, str):
                mask = Image.open(mask)
            mask_l = mask.convert("L")
            mask_tensor = (
                torch.tensor(np.array(mask_l) > 0, dtype=torch.bool)
                .unsqueeze(0)
                .to(self.device)
            )

        mask_tensor = (
            F.interpolate(
                mask_tensor.unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)
            > 0.5
        )

        if self._ref_images is None:
            self._ref_images = img_tensor
            self._ref_masks = mask_tensor
        else:
            self._ref_images = torch.cat([self._ref_images, img_tensor], dim=0)
            self._ref_masks = torch.cat([self._ref_masks, mask_tensor], dim=0)

    def set_target(
        self,
        image: str | Image.Image,
        gt_mask: str | Image.Image | torch.Tensor | None = None,
    ) -> None:
        """Set target image from a file path or PIL Image."""
        _ = gt_mask
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        self._orig_tgt_size = (image.height, image.width)
        self._tgt_image = self._transform(image).to(self.device)

    def segment(self) -> torch.Tensor:
        """Run prediction using previously set reference(s) and target."""
        pred = self.predict(self._ref_images, self._ref_masks, self._tgt_image)
        self._ref_images = None
        self._ref_masks = None
        self._tgt_image = None
        self._orig_tgt_size = None
        return pred

    @torch.no_grad()
    def predict(
        self,
        ref_images: torch.Tensor,
        ref_masks: torch.Tensor,
        tgt_image: torch.Tensor,
    ) -> torch.Tensor:
        """Segment the target image given reference image(s) and mask(s).

        Pipeline: Part1 → Part2 → Part3 → Part4 → binarize → finalize.
        """
        S = ref_images.shape[0]
        tgt_image = tgt_image.unsqueeze(0)
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)

        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, _, h, w = fmaps_norm.shape
        ref_masks = ref_masks.unsqueeze(1)

        # Part 1 — positional debiasing
        fmaps_norm = self._part1_positional_debias(fmaps_norm, ref_masks, S)

        # Part 2 — two-stage background suppression (stage1 gate + stage2 fg/bg score)
        part2 = self._part2_background_suppression(
            fmaps_norm=fmaps_norm,
            ref_masks=ref_masks,
            n_refs=S,
            h=h,
            w=w,
        )
        if part2 is None:
            raise RuntimeError("No foreground tokens in reference mask(s).")
        score, sf, sbn, mu_fg, tgt_feat_denoised = part2

        # Part 3 — clustering (candidates + seed-cluster prior)
        score, cand_soft, seed_prior = self._part3_clustering(
            score,
            sf=sf,
            mu_fg=mu_fg,
            ref_feats_raw=fmaps_norm[:, :S],
            tgt_feat_raw=fmaps_norm[:, S],
            ref_masks=ref_masks,
            n_refs=S,
            h=h,
            w=w,
        )

        # Same-run reference: original Part4 followed by the same P1.2/refiner.
        score_part3 = score
        baseline_part4 = self._part4_semantic_consistency_correction(
            score_part3, sf=sf, sbn=sbn, cand_soft=cand_soft,
            seed_prior=seed_prior, tgt_feat=tgt_feat_denoised,
        )
        baseline_refined = self._p1_uncertainty_gated_anisotropic_diffusion(
            baseline_part4, target_feat=tgt_feat_denoised, target_rgb=tgt_image,
        )
        baseline_mask = self._binarize_response(
            baseline_refined, target_hw=(tgt_image.shape[-2], tgt_image.shape[-1]),
        )
        self.last_hg_a1_baseline_mask = self._finalize_mask(baseline_mask, tgt_image)

        # Active Part4 — sparse signed hypergraph consolidation.
        score = self._part4_hypergraph_consolidation(
            score_part3,
            sf=sf,
            sbn=sbn,
            cand_soft=cand_soft,
            seed_prior=seed_prior,
            tgt_feat=tgt_feat_denoised,
        )
        score_before_p1 = score.clone()
        score = self._p1_uncertainty_gated_anisotropic_diffusion(
            score,
            target_feat=tgt_feat_denoised,
            target_rgb=tgt_image,
        )
        self.last_p1_patch_attribution_state.update({
            "sf": sf.detach(), "sbn": sbn.detach(), "cand_soft": cand_soft.detach(),
            "seed_prior": seed_prior.detach(),
        })
        before_mask = self._binarize_response(
            score_before_p1,
            target_hw=(tgt_image.shape[-2], tgt_image.shape[-1]),
        )
        self.last_p1_before_mask = self._finalize_mask(before_mask, tgt_image)
        p1_2_mask = self._binarize_response(
            self.last_p1_patch_attribution_state["score_p1_2"].squeeze(0),
            target_hw=(tgt_image.shape[-2], tgt_image.shape[-1]),
        )
        self.last_p1_2_mask = self._finalize_mask(p1_2_mask, tgt_image)

        denoised_mask = self._binarize_response(
            score,
            target_hw=(tgt_image.shape[-2], tgt_image.shape[-1]),
        )
        return self._finalize_mask(denoised_mask, tgt_image)

    # ──────────────────────── Shared utilities ────────────────────────

    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T = imgs.shape[:2]
        x = einops.rearrange(imgs, "b t c h w -> (b t) c h w")
        fmaps = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        return einops.rearrange(fmaps, "(b t) c h w -> b t c h w", b=B)

    def _binarize_response(
        self,
        score_hw: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Min-max normalize response, upsample, then threshold."""
        t = 0.5
        score = score_hw - score_hw.min()
        score = score / score.max().clamp_min(1e-6)
        H, W = target_hw
        score = F.interpolate(
            score.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return score > t

    def _finalize_mask(self, mask: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Upsample mask, optional CRF, then original target resolution."""
        H, W = tgt_image.shape[-2:]
        up = mask if mask.shape == (H, W) else upsample_mask(mask, H, W)
        if self.mask_refiner == "crf":
            up = crf_refine(
                self._crf, self._crf_band_px, self._crf_p_core, tgt_image, up
            )
        if self.resize_to_orig_size:
            up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
        return up

    def _p1_uncertainty_gated_anisotropic_diffusion(
        self,
        score_part4: torch.Tensor,
        *,
        target_feat: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """One 4-neighbor, uncertainty-gated anisotropic diffusion step."""
        score_raw = score_part4.unsqueeze(0) if score_part4.ndim == 2 else score_part4
        _, h, w = score_raw.shape
        score_min, score_max = score_raw.amin(), score_raw.amax()
        score_range = score_max - score_min
        score_norm = (score_raw - score_min) / score_range.clamp_min(1e-6)
        feat = F.normalize(target_feat, p=2, dim=1)
        rgb = F.interpolate(
            denormalize(target_rgb).clamp(0.0, 1.0),
            size=(h, w), mode="bilinear", align_corners=False,
        )

        d_feat_lr = (1.0 - (feat[:, :, :, :-1] * feat[:, :, :, 1:]).sum(dim=1)).clamp_min(0.0)
        d_feat_ud = (1.0 - (feat[:, :, :-1, :] * feat[:, :, 1:, :]).sum(dim=1)).clamp_min(0.0)
        d_rgb_lr = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).pow(2).sum(dim=1)
        d_rgb_ud = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).pow(2).sum(dim=1)
        sigma_feat = torch.cat([d_feat_lr.reshape(-1), d_feat_ud.reshape(-1)]).median().clamp_min(1e-6)
        sigma_rgb = torch.cat([d_rgb_lr.reshape(-1), d_rgb_ud.reshape(-1)]).median().clamp_min(1e-6)
        c_lr = torch.exp(-d_feat_lr / sigma_feat - d_rgb_lr / sigma_rgb)
        c_ud = torch.exp(-d_feat_ud / sigma_feat - d_rgb_ud / sigma_rgb)

        weighted_neighbor_sum = torch.zeros_like(score_norm)
        conductance_sum = torch.zeros_like(score_norm)
        weighted_neighbor_sum[:, :, :-1] += c_lr * score_norm[:, :, 1:]
        weighted_neighbor_sum[:, :, 1:] += c_lr * score_norm[:, :, :-1]
        conductance_sum[:, :, :-1] += c_lr
        conductance_sum[:, :, 1:] += c_lr
        weighted_neighbor_sum[:, :-1, :] += c_ud * score_norm[:, 1:, :]
        weighted_neighbor_sum[:, 1:, :] += c_ud * score_norm[:, :-1, :]
        conductance_sum[:, :-1, :] += c_ud
        conductance_sum[:, 1:, :] += c_ud
        neighbor_mean = torch.where(
            conductance_sum > 1e-8,
            weighted_neighbor_sum / conductance_sum.clamp_min(1e-8),
            score_norm,
        )

        confidence = (2.0 * score_norm - 1.0).abs().clamp(0.0, 1.0)
        score_norm_symmetric = confidence * score_norm + (1.0 - confidence) * neighbor_mean
        step1_dissipation = (1.0 - confidence) * (score_norm - neighbor_mean).clamp_min(0.0)
        score_norm_p1_2 = score_norm - step1_dissipation
        def neighbor_for(field):
            weighted = torch.zeros_like(field)
            weighted[:, :, :-1] += c_lr * field[:, :, 1:]; weighted[:, :, 1:] += c_lr * field[:, :, :-1]
            weighted[:, :-1, :] += c_ud * field[:, 1:, :]; weighted[:, 1:, :] += c_ud * field[:, :-1, :]
            return torch.where(conductance_sum > 1e-8, weighted / conductance_sum.clamp_min(1e-8), field)
        neighbor_mean_step2 = neighbor_for(score_norm_p1_2)
        confidence_step2 = (2.0 * score_norm_p1_2 - 1.0).abs().clamp(0.0, 1.0)
        step2_dissipation = (1.0 - confidence_step2) * (score_norm_p1_2 - neighbor_mean_step2).clamp_min(0.0)
        score_norm_step2 = score_norm_p1_2 - step2_dissipation
        neighbor_mean_step3 = neighbor_for(score_norm_step2)
        confidence_step3 = (2.0 * score_norm_step2 - 1.0).abs().clamp(0.0, 1.0)
        step3_dissipation = (1.0 - confidence_step3) * (score_norm_step2 - neighbor_mean_step3).clamp_min(0.0)
        score_norm_p1 = score_norm_step2 - step3_dissipation
        score_p1_1 = score_raw if bool(score_range <= 1e-6) else score_min + score_range * score_norm_p1_2
        score_p1_2 = score_raw if bool(score_range <= 1e-6) else score_min + score_range * score_norm_step2
        score_p1 = score_raw if bool(score_range <= 1e-6) else score_min + score_range * score_norm_p1

        conductance = torch.cat([c_lr.reshape(-1), c_ud.reshape(-1)])
        upward_candidate = score_norm_symmetric > score_norm
        downward_candidate = score_norm_symmetric < score_norm
        rejected_upward = (score_norm_symmetric - score_norm).clamp_min(0.0)
        accepted_dissipation = (score_norm - score_norm_p1).clamp_min(0.0)
        before_patch = score_norm > 0.5
        after_patch = score_norm_p1 > 0.5
        patch_fg_to_bg = before_patch & (~after_patch)
        patch_bg_to_fg = (~before_patch) & after_patch
        monotonicity_violation = score_norm_p1 > score_norm + 1e-7
        if bool(patch_bg_to_fg.any()):
            raise RuntimeError("P1.2 monotonicity violation: patch BG->FG flip detected.")
        self.last_p1_diffusion_analysis = {
            "mode": "p1_2_three_step_one_sided_dissipation",
            "raw_score_min": float(score_min),
            "raw_score_max": float(score_max),
            "raw_score_range": float(score_range),
            "normalized_score_mean": float(score_norm.mean()),
            "normalized_confidence_mean": float(confidence.mean()),
            "normalized_low_conf_fraction": float((confidence < 0.5).float().mean()),
            "score_norm_change_abs_mean": float((score_norm_p1 - score_norm).abs().mean()),
            "score_norm_change_abs_max": float((score_norm_p1 - score_norm).abs().max()),
            "raw_score_change_abs_mean": float((score_p1 - score_raw).abs().mean()),
            "raw_score_change_abs_max": float((score_p1 - score_raw).abs().max()),
            "score_before_mean": float(score_raw.mean()),
            "score_after_mean": float(score_p1.mean()),
            "score_change_abs_mean": float((score_p1 - score_raw).abs().mean()),
            "score_change_abs_max": float((score_p1 - score_raw).abs().max()),
            "confidence_mean": float(confidence.mean()),
            "conductance_mean": float(conductance.mean()),
            "conductance_std": float(conductance.std(unbiased=False)),
            "sigma_feat": float(sigma_feat),
            "sigma_rgb": float(sigma_rgb),
            "neighbor_mean_minus_score_abs_mean": float((neighbor_mean - score_norm).abs().mean()),
            "patch_threshold_flip_count": int((before_patch != after_patch).sum()),
            "patch_threshold_flip_fraction": float((before_patch != after_patch).float().mean()),
            "threshold_flip_count": int((before_patch != after_patch).sum()),
            "threshold_flip_fraction": float((before_patch != after_patch).float().mean()),
            "low_conf_fraction": float((confidence < 0.5).float().mean()),
            "anchor_min_error": float(score_norm_p1.reshape(-1)[score_raw.argmin()].abs()),
            "anchor_max_error": float((score_norm_p1.reshape(-1)[score_raw.argmax()] - 1.0).abs()),
            "upward_candidate_fraction": float(upward_candidate.float().mean()),
            "downward_candidate_fraction": float(downward_candidate.float().mean()),
            "unchanged_candidate_fraction": float((~(upward_candidate | downward_candidate)).float().mean()),
            "upward_candidate_abs_mean": float(rejected_upward.mean()),
            "downward_candidate_abs_mean": float((score_norm - score_norm_symmetric).clamp_min(0.0).mean()),
            "rejected_upward_abs_mean": float(rejected_upward.mean()),
            "rejected_upward_abs_max": float(rejected_upward.max()),
            "accepted_dissipation_mean": float(accepted_dissipation.mean()),
            "accepted_dissipation_max": float(accepted_dissipation.max()),
            "patch_fg_to_bg_count": int(patch_fg_to_bg.sum()),
            "patch_bg_to_fg_count": int(patch_bg_to_fg.sum()),
            "monotonicity_violation_count": int(monotonicity_violation.sum()),
            "monotonicity_violation_max": float((score_norm_p1 - score_norm).clamp_min(0.0).max()),
            "step1_dissipation_mean": float(step1_dissipation.mean()), "step1_dissipation_max": float(step1_dissipation.max()),
            "step2_dissipation_mean": float(step2_dissipation.mean()), "step2_dissipation_max": float(step2_dissipation.max()),
            "step3_dissipation_mean": float(step3_dissipation.mean()), "step3_dissipation_max": float(step3_dissipation.max()),
            "step2_to_step1_dissipation_ratio": None if float(step1_dissipation.mean()) <= 1e-8 else float(step2_dissipation.mean() / step1_dissipation.mean()),
            "step1_monotonicity_violation_count": int((score_norm_p1_2 > score_norm + 1e-7).sum()), "step2_monotonicity_violation_count": int((score_norm_step2 > score_norm_p1_2 + 1e-7).sum()), "step3_monotonicity_violation_count": int((score_norm_p1 > score_norm_step2 + 1e-7).sum()),
            "core_protection_mean_fg": 0.0, "dissipation_reduction_mean": 0.0,
            "p1_2_lower_bound_violation_count": 0,
        }
        self.last_p1_patch_attribution_state = {
            "score_norm": score_norm.detach(), "score_norm_p1": score_norm_p1.detach(),
            "confidence": confidence.detach(), "neighbor_mean_norm": neighbor_mean.detach(),
            "score_norm_p1_2": score_norm_step2.detach(), "score_p1_2": score_p1_2.detach(), "score_p1_1": score_p1_1.detach(),
            "step1_dissipation": step1_dissipation.detach(), "step2_dissipation": step2_dissipation.detach(), "step3_dissipation": step3_dissipation.detach(),
            "fg_neighbor_count": torch.zeros_like(score_norm, dtype=torch.long),
        }
        return score_p1.squeeze(0) if score_part4.ndim == 2 else score_p1

    # ══════════════════════════════════════════════════════════════════════
    # Part 1: Positional debiasing (是否去除位置偏置)
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Estimate the positional subspace from a noise image via SVD."""
        from torchvision.transforms.functional import normalize

        noise_img = normalize(
            torch.zeros(1, 3, self.image_size, self.image_size),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).to(device)
        noise_fmaps = self.encoder.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True
        )[0]
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)

        E = einops.rearrange(noise_fmaps, "b c h w -> c (b h w)")
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, : self.svd_components].contiguous()

    @torch.no_grad()
    def _should_apply_positional_debias(
        self,
        fmaps_feats: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
    ) -> bool:
        """Decide whether to project features off the positional subspace."""
        _, _, C, Hf, Wf = fmaps_feats.shape

        mask_ds = (
            F.interpolate(ref_masks.float(), size=(Hf, Wf), mode="nearest") > 0.5
        ).squeeze(1)

        basis = self.positional_basis.to(
            device=fmaps_feats.device, dtype=fmaps_feats.dtype
        )

        tgt = fmaps_feats[0, -1].reshape(C, -1).mean(dim=1)
        scores: list[torch.Tensor] = []
        s_sem_last = None

        for s in range(n_refs):
            feat = fmaps_feats[0, s].reshape(C, -1)
            mask = mask_ds[s].reshape(-1)
            if mask.sum() == 0:
                continue

            fg = feat[:, mask]
            if fg.shape[1] == 0:
                continue

            mu_fg = F.normalize(fg.mean(dim=1), dim=0)
            s_sem = torch.dot(mu_fg, tgt) / (tgt.norm() + 1e-6)
            s_sem_last = s_sem
            proj = basis @ (basis.T @ tgt)
            s_pos = proj.norm() / (tgt.norm() + 1e-6)
            scores.append(s_pos - s_sem)

        if len(scores) == 0 or s_sem_last is None:
            return True

        self.should_debiass = s_sem_last.item() < 0.8

        return s_sem_last.item() < 0.8

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """Project features onto the orthogonal complement of the positional subspace."""
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)

        basis = self.positional_basis.to(X.device)
        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        return F.normalize(X_deb, p=2, dim=2)

    def _part1_positional_debias(
        self,
        fmaps_norm: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
    ) -> torch.Tensor:
        """Part 1 entry: optionally remove positional subspace bias."""

        # 1 需要可视化fmaps_norm,分别可视化
        if self._should_apply_positional_debias(fmaps_norm, ref_masks, n_refs):
            return self._debias_features(fmaps_norm)
        return fmaps_norm

    # ══════════════════════════════════════════════════════════════════════
    # Part 2: Two-stage background suppression (两阶段去除背景)
    #   Stage 1 — feature gating via fg/bg contrast on all tokens
    #   Stage 2 — pixel-wise fg prototype score minus background penalty
    # ══════════════════════════════════════════════════════════════════════

    def _reference_contrastive_prototypes(
        self,
        ref_feats: torch.Tensor,
        ref_masks_bool: torch.Tensor,
        n_refs: int,
        *,
        mu_fg_per_reference: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Build mu_fg, hard-negative mu_bg, and optional clustered FG prototypes."""
        ref_means: list[torch.Tensor] = []
        fg_cols: list[torch.Tensor] = []
        bg_cols: list[torch.Tensor] = []

        for s in range(n_refs):
            feat_s = ref_feats[0, s]
            m = ref_masks_bool[s]
            fg = feat_s[:, m]
            bg = feat_s[:, ~m]
            if fg.shape[1] > 0:
                fg_cols.append(fg)
                ref_means.append(fg.mean(dim=1))
            if bg.shape[1] > 0:
                bg_cols.append(bg)

        if len(fg_cols) == 0:
            return None

        fg_tokens = torch.cat(fg_cols, dim=1)
        if mu_fg_per_reference:
            mu_fg = F.normalize(torch.stack(ref_means, dim=0).mean(dim=0), p=2, dim=0)
        else:
            mu_fg = F.normalize(fg_tokens.mean(dim=1), dim=0)

        if len(bg_cols) > 0:
            bg_tokens_cat = torch.cat(bg_cols, dim=1)
            sim_bg = torch.einsum("cn,c->n", bg_tokens_cat, mu_fg)
            k_hard = max(1, int(0.2 * sim_bg.numel()))
            topk_idx = torch.topk(sim_bg, k=k_hard).indices
            hard_bg = bg_tokens_cat[:, topk_idx]
            mu_bg = F.normalize(hard_bg.mean(dim=1), dim=0)
        else:
            mu_bg = torch.zeros_like(mu_fg)

        fg_protos: torch.Tensor | None = None
        if self.enable_clustering and fg_tokens.shape[1] > 0:
            x_fg = F.normalize(fg_tokens.transpose(0, 1), p=2, dim=1)
            labels_fg = agglomerative_clustering(x_fg, tau=self.tau)
            k_fg = int(labels_fg.max().item()) + 1
            fg_protos = compute_cluster_prototypes(x_fg, labels_fg, K=k_fg)

        return mu_fg, mu_bg, fg_protos

    def _part2_stage1_feature_gating(
        self,
        fmaps_norm: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
    ) -> torch.Tensor:
        """Part 2 stage 1: contrastive fg/bg gating on feature maps."""
        B, T, C, Hf, Wf = fmaps_norm.shape

        mask_ds = F.interpolate(ref_masks.float(), size=(Hf, Wf), mode="nearest") > 0.5
        mask_ds = mask_ds.squeeze(1)

        stats = self._reference_contrastive_prototypes(
            fmaps_norm[:, :n_refs],
            mask_ds,
            n_refs,
            mu_fg_per_reference=False,
        )
        if stats is None:
            return fmaps_norm
        mu_fg, mu_bg, fg_protos = stats

        if fg_protos is not None and fg_protos.shape[0] > 0:
            sim_fg_k = torch.einsum("btchw,kc->btkhw", fmaps_norm, fg_protos)
            t = max(1e-4, self.cluster_logsumexp_temp)
            score_fg = t * torch.logsumexp(sim_fg_k / t, dim=2)
        else:
            score_fg = torch.einsum("btchw,c->bthw", fmaps_norm, mu_fg)
        score_bg = torch.einsum("btchw,c->bthw", fmaps_norm, mu_bg)
        score = score_fg - score_bg
        score = score - score.mean(dim=(-2, -1), keepdim=True)

        gate = torch.sigmoid(12.0 * score)
        ref_gate_floor = mask_ds.unsqueeze(0).to(dtype=gate.dtype)
        gate[:, :n_refs] = torch.maximum(gate[:, :n_refs], ref_gate_floor)

        min_keep = 0.1
        gate = min_keep + (1.0 - min_keep) * gate
        return fmaps_norm * gate.unsqueeze(2)

    def _part2_stage2_contrastive_score(
        self,
        ref_feats: torch.Tensor,
        tgt_feat: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
        h: int,
        w: int,
        tgt_feat_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Part 2 stage 2: fg prototype matching with hard-negative bg suppression."""
        w_bg = float(self.dino_bg_weight)

        target_feat_for_score = (
            tgt_feat_raw
            if (self.use_raw_target_scoring and tgt_feat_raw is not None)
            else tgt_feat
        )
        target_feat_for_cluster = (
            tgt_feat_raw
            if (self.use_raw_target_clustering and tgt_feat_raw is not None)
            else tgt_feat
        )

        ref_masks_ds = torch.stack(
            [downsample_mask(ref_masks[s : s + 1], h, w) for s in range(n_refs)],
            dim=0,
        )
        stats = self._reference_contrastive_prototypes(
            ref_feats,
            ref_masks_ds,
            n_refs,
            mu_fg_per_reference=True,
        )
        if stats is None:
            return None
        mu_fg, mu_bg, fg_protos = stats

        # Hard-negative mean is often correlated with mu_fg; orthogonalize so sim_bg
        # measures similarity along directions not explained by the foreground prototype.
        dot = (mu_bg * mu_fg).sum()
        orth = mu_bg - dot * mu_fg
        n = orth.norm()
        mu_bg = orth / n.clamp_min(1e-8)

        # Foreground scoring: clustered multi-prototypes via log-sum-exp aggregation.
        sim_khw = torch.einsum(
            "bchw,kc->bkhw", target_feat_for_cluster, fg_protos
        )  # (1, K, h, w)
        t = max(1e-4, self.cluster_logsumexp_temp)
        sim_fg_hw = (t * torch.logsumexp(sim_khw / t, dim=1)).squeeze(0)  # (h, w)

        sim_bg_hw = torch.einsum("bchw,c->bhw", target_feat_for_score, mu_bg)
        sb = sim_bg_hw.squeeze(0)  # (h, w)
        score_dino = sim_fg_hw - w_bg * sb

        sf = sim_fg_hw - sim_fg_hw.min()
        sf = sf / sf.max().clamp_min(1e-6)
        sbn = sb - sb.min()
        sbn = sbn / sbn.max().clamp_min(1e-6)
        return score_dino, sf, sbn, mu_fg

    def _part2_background_suppression(
        self,
        fmaps_norm: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
        h: int,
        w: int,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Part 2 entry: stage-1 feature gating + stage-2 fg/bg contrastive response."""
        fmaps_denoised = self._part2_stage1_feature_gating(
            fmaps_norm, ref_masks, n_refs
        )
        tgt_feat_denoised = fmaps_denoised[:, n_refs]
        ref_feats = fmaps_norm[:, :n_refs]
        tgt_feat_raw = fmaps_norm[:, n_refs]

        stage2 = self._part2_stage2_contrastive_score(
            ref_feats=ref_feats,
            tgt_feat=tgt_feat_denoised,
            ref_masks=ref_masks,
            n_refs=n_refs,
            h=h,
            w=w,
            tgt_feat_raw=tgt_feat_raw,
        )
        if stage2 is None:
            return None
        score, sf, sbn, mu_fg = stage2
        return score, sf, sbn, mu_fg, tgt_feat_denoised

    # ══════════════════════════════════════════════════════════════════════
    # Part 3: Clustering (聚类)
    #   Candidate localization, seed-cluster prior, score boosting
    # ══════════════════════════════════════════════════════════════════════

    def _locate_candidates(
        self,
        ref_feats: torch.Tensor,
        tgt_feat: torch.Tensor,
        ref_masks: torch.Tensor,
        ref_prototype: torch.Tensor,
        n_refs: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if ref_feats.shape[0] != 1 or tgt_feat.shape[0] != 1:
            ones = torch.ones((h, w), device=tgt_feat.device, dtype=torch.bool)
            z = torch.zeros((h, w), device=tgt_feat.device, dtype=tgt_feat.dtype)
            return ones, ones, ones, z, z

        device = tgt_feat.device
        dtype = tgt_feat.dtype
        tgt_norm = F.normalize(tgt_feat, p=2, dim=1)

        # Soft top-k foreground evidence: preserve the original affinity and
        # normalization path, changing only top-1 binary voting.
        soft_sum = torch.zeros((h, w), dtype=dtype, device=device)
        temperature = max(1e-4, float(self.cluster_logsumexp_temp))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        yy, xx = yy.unsqueeze(-1), xx.unsqueeze(-1)
        mutual_radius = 1
        for m in range(n_refs):
            ref_m = F.normalize(ref_feats[0:1, m], p=2, dim=2)
            sim_m = torch.einsum("bchw,bcxy->bhwxy", ref_m, tgt_norm)
            sim0 = sim_m[0]
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)
            sim_flat = sim_t_to_r.reshape(h, w, -1)
            k_eff = min(5, Hs * Ws)
            topk_vals, topk_idx = torch.topk(sim_flat, k=k_eff, dim=-1)
            ref_mask_m = downsample_mask(ref_masks[m : m + 1], Hs, Ws).squeeze(0)
            topk_mask = ref_mask_m.reshape(-1)[topk_idx].to(dtype=dtype)
            weights = torch.softmax(topk_vals / temperature, dim=-1)
            support_soft_c1 = (weights * topk_mask).sum(dim=-1)

            # Reuse sim0 for a single reverse argmax; no second affinity is made.
            back_best_idx = sim0.reshape(Hs * Ws, h * w).argmax(dim=-1)
            topk_back_idx = back_best_idx[topk_idx]
            back_rows, back_cols = topk_back_idx // w, topk_back_idx % w
            mutual_mask = (
                (back_rows - yy).abs() <= mutual_radius
            ) & ((back_cols - xx).abs() <= mutual_radius)
            mutual_weights = weights * mutual_mask.to(dtype=weights.dtype)
            mutual_mass = mutual_weights.sum(dim=-1, keepdim=True)
            weights_mutual = mutual_weights / mutual_mass.clamp_min(1e-8)
            support_soft_mutual = (weights_mutual * topk_mask).sum(dim=-1)
            support_soft = torch.where(
                mutual_mass.squeeze(-1) > 1e-8,
                support_soft_mutual,
                support_soft_c1,
            )
            soft_sum += support_soft

        vote_soft = (soft_sum / float(max(1, n_refs))).clamp(0.0, 1.0)
        candidate_threshold = math.ceil(n_refs / 2) / float(max(1, n_refs))
        candidates_mask = vote_soft >= candidate_threshold

        return candidates_mask,  vote_soft

    def  _build_seed_cluster_prior(
        self,
        ref_feats: torch.Tensor,
        tgt_feat: torch.Tensor,
        ref_masks: torch.Tensor,
        candidate_mask: torch.Tensor,
        n_refs: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Soft prior from target seed-cluster aggregation."""
        if ref_feats.shape[0] != 1 or tgt_feat.shape[0] != 1:
            return torch.ones((h, w), dtype=tgt_feat.dtype, device=tgt_feat.device) * 0.5

        feat_tgt = tgt_feat[0]
        c = feat_tgt.shape[0]

        x = feat_tgt.reshape(c, -1).transpose(0, 1)
        x = F.normalize(x, p=2, dim=1)
        x_cluster = x

        if self._tgt_image is not None:
            tgt_low = F.interpolate(
                self._tgt_image.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]
            tgt_rgb = denormalize(tgt_low).clamp(0.0, 1.0)
            color_tokens = tgt_rgb.reshape(3, -1).transpose(0, 1)
            color_tokens = F.normalize(color_tokens, p=2, dim=1)

            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype),
                torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            pos_tokens = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
            pos_tokens = F.normalize(pos_tokens, p=2, dim=1)

            w_feat, w_color, w_pos = 1.00, 0.35, 0.20
            x_cluster = torch.cat(
                [x * w_feat, color_tokens * w_color, pos_tokens * w_pos],
                dim=1,
            )
            x_cluster = F.normalize(x_cluster, p=2, dim=1)

        labels = agglomerative_clustering(x_cluster, tau=self.tau)
        k = int(labels.max().item()) + 1
        if k <= 1:
            return candidate_mask.to(dtype=tgt_feat.dtype)

        protos = compute_cluster_prototypes(x, labels, K=k)

        ref_protos = []
        for s in range(n_refs):
            mask_s = downsample_mask(ref_masks[s : s + 1], h, w)
            fg_s = ref_feats[0, s][:, mask_s]
            if fg_s.shape[1] > 0:
                ref_protos.append(fg_s.mean(dim=1))
        if len(ref_protos) == 0:
            return candidate_mask.to(dtype=tgt_feat.dtype)

        mu_fg = F.normalize(torch.stack(ref_protos).mean(dim=0), p=2, dim=0)

        labels_hw = labels.view(h, w)
        matched = labels_hw[candidate_mask]
        if matched.numel() == 0:
            return candidate_mask.to(dtype=tgt_feat.dtype)

        matched_ids, counts = matched.unique(return_counts=True)

        area_all = torch.bincount(labels, minlength=k).to(dtype=protos.dtype).clamp_min(1.0)
        area_w = torch.zeros(k, device=protos.device, dtype=protos.dtype)
        area_w[matched_ids] = counts.to(dtype=protos.dtype)
        area_w = area_w / area_all

        fg_sim_map = torch.einsum("chw,c->hw", feat_tgt, mu_fg)
        cross_sim = torch.zeros(k, device=protos.device, dtype=protos.dtype)
        labels_flat = labels.view(-1)
        fg_sim_flat = fg_sim_map.view(-1)
        for i in range(k):
            mask_i = labels_flat == i
            if mask_i.any():
                cross_sim[i] = fg_sim_flat[mask_i].mean()

        seed_scores = cross_sim * area_w
        seed_cluster = int(matched_ids[torch.argmax(seed_scores[matched_ids])].item())

        intra_sim = torch.einsum("c,kc->k", protos[seed_cluster], protos).clamp_min(0.0)
        combined = cross_sim * intra_sim * area_w
        combined[seed_cluster] = combined.max().clamp_min(1e-6)

        lo, hi = combined.min(), combined.max()
        combined = (combined - lo) / (hi - lo).clamp_min(1e-6)
        return combined[labels].view(h, w)

    def _part3_clustering(
        self,
        score: torch.Tensor,
        *,
        sf: torch.Tensor,
        mu_fg: torch.Tensor,
        ref_feats_raw: torch.Tensor,
        tgt_feat_raw: torch.Tensor,
        ref_masks: torch.Tensor,
        n_refs: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Part 3 entry: candidate localization + seed-cluster prior boosts."""
        candidate_boost = float(self.candidate_boost)
        seed_cluster_boost = float(self.seed_cluster_boost)

        # Reference soft maps for Part 4 semantic disagreement (initialized before boosts).

        seed_prior_ref = torch.full_like(sf, 0.5)

        # candidate localization on raw target features.
        cand_base,  vote_soft = self._locate_candidates(
            ref_feats=ref_feats_raw,
            tgt_feat=tgt_feat_raw,
            ref_masks=ref_masks,
            ref_prototype=mu_fg,
            n_refs=n_refs,
            h=h,
            w=w,
        )

        ratio = 0.25
        step = max(1, round((1 / ratio) ** 0.5))
        densify_mask = torch.zeros(
            (h, w),
            dtype=torch.bool,
            device=sf.device,
        )
        densify_mask[::step, ::step] = True
        cand = cand_base | densify_mask

        cand_soft = vote_soft
        score = score + candidate_boost * (cand_soft - 0.5)
        cand_soft_ref = cand_soft

        prior = self._build_seed_cluster_prior(
            ref_feats=ref_feats_raw,
            tgt_feat=tgt_feat_raw,
            ref_masks=ref_masks,
            candidate_mask=cand,
            n_refs=n_refs,
            h=h,
            w=w,
        )
        # print(prior)
      
        score = score + seed_cluster_boost * (prior - 0.5)
        seed_prior_ref = prior

        # Return final references used by Part 4 (equivalent to cand_soft_ref / seed_prior_ref).
        return score, cand_soft_ref, seed_prior_ref

    # ══════════════════════════════════════════════════════════════════════
    # Part 4: Semantic consistency correction (语义一致性修正)
    # ══════════════════════════════════════════════════════════════════════

    def _semantic_disagreement_penalty(
        self,
        sf: torch.Tensor,
        sbn: torch.Tensor,
        cand_soft: torch.Tensor,
        seed_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize regions where semantic evidences disagree or fg/bg are coupled."""
        disagree = (sf - cand_soft).abs() + (sf - seed_prior).abs()
        fg_bg_coupling = torch.minimum(sf, sbn)
        penalty = (
            self.semantic_disagreement_weight * disagree
            + self.semantic_bg_coupling_weight * fg_bg_coupling
        )

        uncertainty = (1.0 - (2.0 * (sf - 0.5).abs())).clamp(0.0, 1.0)

        penalty = penalty * (uncertainty ** self.semantic_penalty_uncertainty_power)
        return penalty.clamp_max(self.semantic_penalty_max)

    def _semantic_cluster_reweight_map(
        self,
        tgt_feat: torch.Tensor,
        sf: torch.Tensor,
        sbn: torch.Tensor,
        cand_soft: torch.Tensor,
        seed_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Cluster-level boost for pure-fg clusters, suppress conflicted clusters."""
        _, c_t, h_t, w_t = tgt_feat.shape
        xt = tgt_feat.squeeze(0).permute(1, 2, 0).reshape(h_t * w_t, c_t)
        xt = F.normalize(xt, p=2, dim=1)
        labels_t = agglomerative_clustering(xt, tau=self.tau)
        k_t = int(labels_t.max().item()) + 1

        sf_flat, sb_flat = sf.reshape(-1), sbn.reshape(-1)
        cand_flat, seed_flat = cand_soft.reshape(-1), seed_prior.reshape(-1)
        delta_cluster = torch.zeros(k_t, device=sf.device, dtype=sf.dtype)
        cluster_sf = torch.zeros_like(delta_cluster); cluster_sb = torch.zeros_like(delta_cluster)
        cluster_cand = torch.zeros_like(delta_cluster); cluster_seed = torch.zeros_like(delta_cluster)

        for k_idx in range(k_t):
            mk = labels_t == k_idx
            if not bool(mk.any()):
                continue
            fg_mean = sf_flat[mk].mean()
            bg_mean = sb_flat[mk].mean()
            cluster_sf[k_idx], cluster_sb[k_idx] = fg_mean, bg_mean
            cluster_cand[k_idx], cluster_seed[k_idx] = cand_flat[mk].mean(), seed_flat[mk].mean()
            fg_pure = (fg_mean - bg_mean).clamp_min(0.0)
            conflict = torch.minimum(fg_mean, bg_mean)
            delta_k = (
                self.semantic_cluster_fg_boost * fg_pure
                - self.semantic_cluster_conflict_suppress * conflict
            )
            delta_cluster[k_idx] = delta_k.clamp_min(-self.semantic_cluster_neg_cap)
        H = torch.stack([(2*(cluster_sf-.5)).clamp(0,1), (2*(cluster_cand-.5)).clamp(0,1), (2*(cluster_seed-.5)).clamp(0,1)], 1)
        degrees = H.sum(0); valid = degrees > 1e-8
        if bool(valid.any()):
            Hv = H[:, valid]; ed = Hv.sum(0).clamp_min(1e-8); nd = Hv.sum(1).clamp_min(1e-8)
            base = (Hv / ed.unsqueeze(0)) @ Hv.T; inv = nd.rsqrt()
            theta = inv.unsqueeze(1) * base * inv.unsqueeze(0)
            direct = Hv.mean(1); group = theta @ direct
            raw_reliability = (.5 * direct + .5 * group).clamp(0, 1)
            gate = (.5 + .5 * raw_reliability).clamp(.5, 1)
            formula_error = float((gate - (.5 + .5 * raw_reliability).clamp(.5, 1)).abs().max())
        else:
            direct = group = raw_reliability = torch.zeros_like(delta_cluster); gate = torch.ones_like(delta_cluster); formula_error = 0.0
        final = gate * delta_cluster
        pos, neg = delta_cluster > 0, delta_cluster < 0
        mean_if = lambda x, m: float(x[m].abs().mean()) if bool(m.any()) else 0.0
        base_sum = delta_cluster.abs().sum()
        pos_ret = float(final[pos].abs().sum() / delta_cluster[pos].abs().sum().clamp_min(1e-8)) if bool(pos.any()) else 1.0
        neg_ret = float(final[neg].abs().sum() / delta_cluster[neg].abs().sum().clamp_min(1e-8)) if bool(neg.any()) else 1.0
        self.last_hg_part4_analysis = {
            "enabled": True, "mode": "original_symmetric_hg_gate", "num_clusters": int(k_t), "num_valid_hyperedges": int(valid.sum()),
            "edge_degree_sf": float(degrees[0]), "edge_degree_cand": float(degrees[1]), "edge_degree_seed": float(degrees[2]),
            "cluster_sf_mean": float(cluster_sf.mean()), "cluster_cand_mean": float(cluster_cand.mean()), "cluster_seed_mean": float(cluster_seed.mean()), "cluster_sbn_mean": float(cluster_sb.mean()),
            "direct_support_mean": float(direct.mean()), "group_support_mean": float(group.mean()),
            "raw_reliability_mean": float(raw_reliability.mean()), "raw_reliability_std": float(raw_reliability.std(unbiased=False)), "raw_reliability_min": float(raw_reliability.min()), "raw_reliability_max": float(raw_reliability.max()), "gate_formula_error_abs_max": formula_error, "empty_hypergraph_fallback": not bool(valid.any()),
            "hg_gate_mean": float(gate.mean()), "hg_gate_std": float(gate.std(unbiased=False)), "hg_gate_min": float(gate.min()), "hg_gate_max": float(gate.max()),
            "delta_base_abs_mean": float(delta_cluster.abs().mean()), "delta_final_abs_mean": float(final.abs().mean()), "delta_change_abs_mean": float((final-delta_cluster).abs().mean()),
            "positive_delta_clusters": int(pos.sum()), "negative_delta_clusters": int(neg.sum()),
            "positive_delta_abs_before": mean_if(delta_cluster,pos), "positive_delta_abs_after": mean_if(final,pos), "negative_delta_abs_before": mean_if(delta_cluster,neg), "negative_delta_abs_after": mean_if(final,neg),
            "correction_retention_ratio": 1.0 if float(base_sum) <= 1e-8 else float(final.abs().sum()/base_sum),
            "sign_flip_count": int(((torch.sign(delta_cluster)!=torch.sign(final)) & (delta_cluster!=0) & (final!=0)).sum()),
            "multi_evidence_cluster_fraction": float(((H>0).sum(1)>=2).float().mean()),
            "positive_retention_ratio": pos_ret, "negative_retention_ratio": neg_ret,
            "positive_delta_change_abs_mean": float((final[pos]-delta_cluster[pos]).abs().mean()) if bool(pos.any()) else 0.0,
            "negative_delta_change_abs_mean": float((final[neg]-delta_cluster[neg]).abs().mean()) if bool(neg.any()) else 0.0,
            "num_positive_gated_clusters": int(pos.sum()), "num_negative_gated_clusters": int(neg.sum()),
        }
        return final[labels_t].view(h_t, w_t)

    def _part4_semantic_consistency_correction(
        self,
        score: torch.Tensor,
        *,
        sf: torch.Tensor,
        sbn: torch.Tensor,
        cand_soft: torch.Tensor,
        seed_prior: torch.Tensor,
        tgt_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Part 4 entry: disagreement penalty + cluster-level semantic reweight."""
        penalty = self._semantic_disagreement_penalty(sf, sbn, cand_soft, seed_prior)
        score = score - penalty
        score = score 
        delta_map = self._semantic_cluster_reweight_map(tgt_feat, sf, sbn, cand_soft, seed_prior)
        return score + delta_map

    def _part4_hypergraph_consolidation(
        self,
        score: torch.Tensor,
        *,
        sf: torch.Tensor,
        sbn: torch.Tensor,
        cand_soft: torch.Tensor,
        seed_prior: torch.Tensor,
        tgt_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Training-free, signed group correction on existing target clusters."""
        _, channels, height, width = tgt_feat.shape
        tokens = F.normalize(
            tgt_feat[0].permute(1, 2, 0).reshape(-1, channels), p=2, dim=1,
        )
        labels = agglomerative_clustering(tokens, tau=self.tau)
        num_nodes = int(labels.max().item()) + 1
        prototypes = compute_cluster_prototypes(tokens, labels, K=num_nodes)
        counts = torch.bincount(labels, minlength=num_nodes).to(dtype=score.dtype).clamp_min(1)

        def cluster_mean(values: torch.Tensor) -> torch.Tensor:
            pooled = torch.zeros(num_nodes, device=score.device, dtype=score.dtype)
            pooled.index_add_(0, labels, values.reshape(-1))
            return pooled / counts

        fg = cluster_mean(sf)
        bg = cluster_mean(sbn)
        candidate = cluster_mean(cand_soft)
        seed = cluster_mean(seed_prior)
        conflict = torch.minimum(fg, bg)
        uncertainty = (1.0 - 2.0 * (fg - 0.5).abs()).clamp(0.0, 1.0)
        disagreement = (fg - candidate).abs() + (fg - seed).abs()
        unary = (
            self.semantic_cluster_fg_boost * (fg - bg).clamp_min(0.0)
            - self.semantic_cluster_conflict_suppress * conflict
            - (self.semantic_disagreement_weight * disagreement
               + self.semantic_bg_coupling_weight * conflict)
            * uncertainty.pow(self.semantic_penalty_uncertainty_power)
        ).clamp(
            min=-(self.semantic_cluster_neg_cap + self.semantic_penalty_max),
            max=self.semantic_cluster_fg_boost,
        )

        # Candidate groups have 3–5 members. A group is selected by evidence
        # and semantic similarity; conflict groups also require spatial adjacency.
        evidence = {
            "fg": (2.0 * (fg - 0.5)).clamp(0.0, 1.0),
            "candidate": (2.0 * (candidate - 0.5)).clamp(0.0, 1.0),
            "seed": (2.0 * (seed - 0.5)).clamp(0.0, 1.0),
            "conflict": (bg * (1.0 - (fg + candidate + seed) / 3.0)).clamp(0.0, 1.0),
        }
        similarity = prototypes @ prototypes.T
        labels_hw = labels.view(height, width)
        adjacent = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=labels.device)
        for left, right in (
            (labels_hw[:, :-1].reshape(-1), labels_hw[:, 1:].reshape(-1)),
            (labels_hw[:-1, :].reshape(-1), labels_hw[1:, :].reshape(-1)),
        ):
            adjacent[left, right] = True
            adjacent[right, left] = True

        groups: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        group_counts = {key: 0 for key in evidence}
        for kind, strength in evidence.items():
            eligible = strength > 0.0
            seen: set[tuple[int, ...]] = set()
            anchor_budget = min(num_nodes, max(1, num_nodes // 4))
            anchor_ids = torch.topk(strength, k=anchor_budget).indices
            for anchor in anchor_ids.tolist():
                if not bool(eligible[anchor]):
                    continue
                pool = eligible.clone()
                pool[anchor] = False
                if kind == "conflict":
                    pool &= adjacent[anchor]
                choices = torch.nonzero(pool, as_tuple=False).flatten()
                if choices.numel() < 2:
                    continue
                ranked = torch.topk(similarity[anchor, choices], k=min(4, choices.numel())).indices
                members = torch.cat((choices.new_tensor([anchor]), choices[ranked]))
                identity = tuple(sorted(members.tolist()))
                if identity in seen:
                    continue
                seen.add(identity)
                groups.append((kind, members, strength[members]))
                group_counts[kind] += 1

        positive_sum = torch.zeros_like(unary)
        negative_sum = torch.zeros_like(unary)
        positive_weight = torch.zeros_like(unary)
        negative_weight = torch.zeros_like(unary)
        reliability_sum = torch.zeros_like(unary)
        incidence_sum = torch.zeros_like(unary)
        edge_reliabilities: list[torch.Tensor] = []
        for kind, members, membership in groups:
            values = unary[members]
            center = values.median()
            spread = (values - center).abs().median()
            amplitude = values.abs().median()
            reliability = (1.0 - spread / (amplitude + 1e-8)).clamp(0.0, 1.0)
            edge_reliabilities.append(reliability)
            weight = membership * reliability
            incidence_sum.index_add_(0, members, membership)
            reliability_sum.index_add_(0, members, weight)
            if kind == "conflict":
                message = (-values).clamp_min(0.0).median()
                negative_sum.index_add_(0, members, weight * message)
                negative_weight.index_add_(0, members, weight)
            else:
                message = values.clamp_min(0.0).median()
                positive_sum.index_add_(0, members, weight * message)
                positive_weight.index_add_(0, members, weight)

        positive = positive_sum / positive_weight.clamp_min(1e-8)
        negative = negative_sum / negative_weight.clamp_min(1e-8)
        node_reliability = (reliability_sum / incidence_sum.clamp_min(1e-8)).clamp(0.0, 1.0)
        correction = (
            (1.0 - node_reliability) * unary
            + node_reliability * (positive - negative)
        ).clamp(
            min=-(self.semantic_cluster_neg_cap + self.semantic_penalty_max),
            max=self.semantic_cluster_fg_boost,
        )
        correction = torch.where(torch.isfinite(correction), correction, unary)
        self.last_hg_a1_analysis = {
            "mode": "a1_sparse_signed_hypergraph_consolidation",
            "num_nodes": num_nodes,
            "num_hyperedges": len(groups),
            "hyperedge_counts": group_counts,
            "mean_hyperedge_size": float(sum(len(m) for _, m, _ in groups) / len(groups)) if groups else 0.0,
            "max_hyperedge_size": max((len(m) for _, m, _ in groups), default=0),
            "isolated_node_fraction": float((incidence_sum <= 1e-8).float().mean()),
            "group_reliability_mean": float(torch.stack(edge_reliabilities).mean()) if groups else None,
            "unary_positive_count": int((unary > 0).sum()),
            "unary_negative_count": int((unary < 0).sum()),
            "correction_positive_count": int((correction > 0).sum()),
            "correction_negative_count": int((correction < 0).sum()),
            "unary_abs_mean": float(unary.abs().mean()),
            "correction_abs_mean": float(correction.abs().mean()),
            "positive_message_mean": float(positive.mean()),
            "negative_message_mean": float(negative.mean()),
            "fallback_node_count": int((reliability_sum <= 1e-8).sum()),
        }
        return score + correction[labels].view(height, width)

