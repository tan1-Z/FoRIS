"""FoRIS inference script (legacy logging entry point)."""

import argparse
import datetime
import json
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_foris_from_args
from utils.metrics import Evaluator, AverageMeter
from utils.visualization import save_episode_visualizations


def main(args: argparse.Namespace) -> float:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_foris_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    start_time = time.time()
    miou = evaluate(args, model, log_file)
    print(f'Total inference time: {time.time() - start_time:.1f}s')
    return miou


def evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> float:
    # ──────── Dataset and loader setup ────────
    ds = build_dataset(args.dataset, args=args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    # Original FoRIS metrics implementations differ on whether ``device`` is
    # accepted; their default CUDA buffers match the supported GPU workflow.
    meter = AverageMeter(args.dataset, list(ds.class_ids))
    meter_hg_a1_baseline = AverageMeter(args.dataset, list(ds.class_ids))
    meter_before = AverageMeter(args.dataset, list(ds.class_ids))
    meter_p1_2 = AverageMeter(args.dataset, list(ds.class_ids))
    component_records = []
    signal_names = ("sf", "sbn", "semantic_margin", "cand_soft", "seed_prior", "candidate_seed_support", "fg_support_mean", "fg_vs_bg_margin", "score_norm", "confidence", "neighbor_mean_norm", "dissipation_gap", "distance_to_threshold")
    pooled_signals = {key: {"corrective": [], "harmful": [], "episode_diffs": []} for key in signal_names}

    def distribution(values):
        if values.numel() == 0:
            return {"count": 0, "mean": None, "std": None, "median": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
        values = values.detach().float().cpu()
        return {"count": int(values.numel()), "mean": float(values.mean()), "std": float(values.std(unbiased=False)), "median": float(values.median()), "p10": float(torch.quantile(values, .10)), "p25": float(torch.quantile(values, .25)), "p50": float(torch.quantile(values, .50)), "p75": float(torch.quantile(values, .75)), "p90": float(torch.quantile(values, .90))}

    def patch_attribution(state, gt):
        score_norm = state["score_norm"].squeeze()
        score_norm_p1 = state["score_norm_p1"].squeeze()
        gt_patch = F.interpolate(gt[None, None].float(), size=score_norm.shape, mode="nearest")[0, 0] > .5
        removed = (score_norm > .5) & ~(score_norm_p1 > .5)
        p1_2_patch = state["score_norm_p1_2"].squeeze() > .5
        before_patch = score_norm > .5
        p1_2_removed = before_patch & ~p1_2_patch
        prevented_patch = p1_2_removed & (score_norm_p1 > .5)
        corrective, harmful = removed & ~gt_patch, removed & gt_patch
        sf, sbn, cand, seed = (state[key].squeeze() for key in ("sf", "sbn", "cand_soft", "seed_prior"))
        signals = {"sf": sf, "sbn": sbn, "semantic_margin": sf - sbn, "cand_soft": cand, "seed_prior": seed, "candidate_seed_support": .5 * (cand + seed), "fg_support_mean": (sf + cand + seed) / 3., "score_norm": score_norm, "confidence": state["confidence"].squeeze(), "neighbor_mean_norm": state["neighbor_mean_norm"].squeeze(), "dissipation_gap": (score_norm - state["neighbor_mean_norm"].squeeze()).clamp_min(0.), "distance_to_threshold": (score_norm - .5).abs()}
        signals["fg_vs_bg_margin"] = signals["fg_support_mean"] - sbn
        fg_neighbor_count = state["fg_neighbor_count"].squeeze()
        neighbor_table = {}
        for k in range(5):
            selected = p1_2_removed & (fg_neighbor_count == k)
            count = int(selected.sum())
            neighbor_table[str(k)] = {"p1_2_removed_count": count, "gt_fg_fraction": float(gt_patch[selected].float().mean()) if count else None, "gt_bg_fraction": float((~gt_patch[selected]).float().mean()) if count else None, "p1_3_prevented_count": int((selected & prevented_patch).sum())}
        result = {"removed_patch_count": int(removed.sum()), "corrective_removed_patch_count": int(corrective.sum()), "harmful_removed_patch_count": int(harmful.sum()), "p1_2_patch_fg_to_bg_count": int(p1_2_removed.sum()), "p1_3_patch_fg_to_bg_count": int(removed.sum()), "prevented_patch_fg_to_bg_count": int(prevented_patch.sum()), "prevented_patch_fg_to_bg_fraction": float(prevented_patch.float().mean()), "prevented_patch_true_fg": int((prevented_patch & gt_patch).sum()), "prevented_patch_false_fg": int((prevented_patch & (~gt_patch)).sum()), "neighbor_count_attribution": neighbor_table, "signals": {}}
        for key, values in signals.items():
            c, h = values[corrective], values[harmful]
            c_stats, h_stats = distribution(c), distribution(h)
            result["signals"][key] = {"corrective": c_stats, "harmful": h_stats}
            if c.numel() and h.numel():
                c_np, h_np = c.detach().float().cpu().numpy(), h.detach().float().cpu().numpy()
                pooled_signals[key]["corrective"].append(c_np)
                pooled_signals[key]["harmful"].append(h_np)
                pooled_signals[key]["episode_diffs"].append(float(h.mean() - c.mean()))
        return result

    # ──────── Evaluation loop ────────
    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):

        ref_imgs = batch['ref_imgs']    # list of PIL Images
        ref_masks = batch['ref_masks']  # list of tensors
        tgt_img = batch['tgt_img']      # PIL Image
        tgt_mask = batch['tgt_mask']    # tensor

        # Set all references
        model._ref_images = None  # Ensure reset
        model._ref_masks = None
        for i in range(len(ref_imgs)):
            model.set_reference(ref_imgs[i], ref_masks[i])
        # Set target (pass GT mask for dataflow PNG / debug when dataflow_image_path is set)
        model.set_target(tgt_img, gt_mask=tgt_mask)
        # Segment
        pred_mask = model.segment()
        pm = pred_mask.detach()
        while pm.dim() > 2:
            pm = pm.squeeze(0)
        pred_hw = (pm.shape[0], pm.shape[1])

        tgt_mask = F.interpolate(
            tgt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_hw, mode='nearest',
        ).squeeze(0).squeeze(0).to(pred_mask.device) > 0.5

        tgt_ignore_idx = batch.get('tgt_ignore_idx')
        if tgt_ignore_idx is not None:
            tgt_ignore_idx = F.interpolate(
                tgt_ignore_idx.unsqueeze(0).unsqueeze(0).float(),
                size=pred_hw, mode='nearest',
            ).squeeze(0).squeeze(0).to(pred_mask.device) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=tgt_ignore_idx,
        )
        class_id = torch.as_tensor(batch['class_id'], device=args.device).reshape(-1)
        meter.update(area_inter, area_union, class_id)
        hg_a1_baseline_mask = getattr(model, "last_hg_a1_baseline_mask", None)
        if hg_a1_baseline_mask is None:
            raise RuntimeError("A1 same-run Part4 baseline mask was not produced.")
        hg_baseline_inter, hg_baseline_union = Evaluator.classify_prediction(
            hg_a1_baseline_mask, tgt_mask, tgt_ignore_idx=tgt_ignore_idx,
        )
        meter_hg_a1_baseline.update(hg_baseline_inter, hg_baseline_union, class_id)

        before_mask = getattr(model, "last_p1_before_mask", None)
        if before_mask is None:
            raise RuntimeError("P1.1 counterfactual mask was not produced.")
        before_inter, before_union = Evaluator.classify_prediction(
            before_mask, tgt_mask, tgt_ignore_idx=tgt_ignore_idx,
        )
        meter_before.update(before_inter, before_union, class_id)
        p1_2_mask = getattr(model, "last_p1_2_mask", None)
        if p1_2_mask is None:
            raise RuntimeError("P1.3 P1.2 counterfactual mask was not produced.")
        p1_2_inter, p1_2_union = Evaluator.classify_prediction(
            p1_2_mask, tgt_mask, tgt_ignore_idx=tgt_ignore_idx,
        )
        meter_p1_2.update(p1_2_inter, p1_2_union, class_id)

        fg_union = area_union[1].clamp_min(1.0)
        episode_iou = (area_inter[1] / fg_union * 100.0).item()
        hg_baseline_iou = (hg_baseline_inter[1] / hg_baseline_union[1].clamp_min(1.0) * 100.0).item()
        hg_baseline_fg, hg_active_fg = hg_a1_baseline_mask.bool(), pred_mask.bool()
        hg_valid = torch.ones_like(hg_active_fg, dtype=torch.bool) if tgt_ignore_idx is None else ~tgt_ignore_idx
        hg_fg_to_bg = hg_baseline_fg & ~hg_active_fg & hg_valid
        hg_bg_to_fg = ~hg_baseline_fg & hg_active_fg & hg_valid
        hg_a1_comparison = {
            "iou_baseline": float(hg_baseline_iou), "iou_active": float(episode_iou),
            "delta_iou": float(episode_iou - hg_baseline_iou),
            "fg_to_bg_count": int(hg_fg_to_bg.sum()), "bg_to_fg_count": int(hg_bg_to_fg.sum()),
            "corrective_flip_count": int((hg_fg_to_bg & ~tgt_mask).sum() + (hg_bg_to_fg & tgt_mask).sum()),
            "harmful_flip_count": int((hg_fg_to_bg & tgt_mask).sum() + (hg_bg_to_fg & ~tgt_mask).sum()),
        }
        before_iou = (before_inter[1] / before_union[1].clamp_min(1.0) * 100.0).item()
        p1_2_iou = (p1_2_inter[1] / p1_2_union[1].clamp_min(1.0) * 100.0).item()
        before_fg, after_fg = before_mask.bool(), pred_mask.bool()
        valid = torch.ones_like(after_fg, dtype=torch.bool) if tgt_ignore_idx is None else ~tgt_ignore_idx
        bg_to_fg = (~before_fg) & after_fg & valid
        fg_to_bg = before_fg & (~after_fg) & valid
        corrective_bg_to_fg = int((bg_to_fg & tgt_mask).sum())
        harmful_bg_to_fg = int((bg_to_fg & (~tgt_mask)).sum())
        corrective_fg_to_bg = int((fg_to_bg & (~tgt_mask)).sum())
        harmful_fg_to_bg = int((fg_to_bg & tgt_mask).sum())
        flip_count = int((bg_to_fg | fg_to_bg).sum())
        valid_count = int(valid.sum())
        corrective_flip_count = corrective_bg_to_fg + corrective_fg_to_bg
        harmful_flip_count = harmful_bg_to_fg + harmful_fg_to_bg
        fg_to_bg_count, bg_to_fg_count = int(fg_to_bg.sum()), int(bg_to_fg.sum())
        p1_2_removed = before_fg & (~p1_2_mask.bool()) & valid
        p1_3_removed = before_fg & (~after_fg) & valid
        prevented_final = p1_2_removed & after_fg
        counterfactual = {"iou_before": float(before_iou), "iou_p1_2": float(p1_2_iou), "iou_after": float(episode_iou), "delta_iou": float(episode_iou - before_iou), "delta_p1_2_vs_before": float(p1_2_iou - before_iou), "delta_p1_3_vs_before": float(episode_iou - before_iou), "delta_p1_3_vs_p1_2": float(episode_iou - p1_2_iou), "flip_count": flip_count, "flip_fraction": float(flip_count / max(1, valid_count)), "fg_to_bg_count": fg_to_bg_count, "bg_to_fg_count": bg_to_fg_count, "p1_2_final_fg_to_bg_count": int(p1_2_removed.sum()), "p1_3_final_fg_to_bg_count": int(p1_3_removed.sum()), "prevented_final_fg_to_bg_count": int(prevented_final.sum()), "protected_true_fg_count": int((prevented_final & tgt_mask).sum()), "protected_false_fg_count": int((prevented_final & (~tgt_mask)).sum()), "fg_to_bg_fraction": float(fg_to_bg_count / max(1, valid_count)), "bg_to_fg_fraction": float(bg_to_fg_count / max(1, valid_count)), "corrective_flip_count": corrective_flip_count, "harmful_flip_count": harmful_flip_count, "corrective_flip_fraction": float(corrective_flip_count / max(1, flip_count)), "harmful_flip_fraction": float(harmful_flip_count / max(1, flip_count)), "net_corrective_flips": corrective_flip_count - harmful_flip_count, "corrective_bg_to_fg": corrective_bg_to_fg, "harmful_bg_to_fg": harmful_bg_to_fg, "corrective_fg_to_bg": corrective_fg_to_bg, "harmful_fg_to_bg": harmful_fg_to_bg}
        analysis = getattr(model, "last_hg_part4_analysis", None)
        hg_a1_analysis = getattr(model, "last_hg_a1_analysis", None)
        p1_analysis = getattr(model, "last_p1_diffusion_analysis", None)
        attribution = None  # P1.3-only topology attribution is inactive for P1.2 2-step.
        if analysis is not None:
            component_records.append({"episode_index": int(idx), "class_id": int(class_id[0].item()),
                                      "num_shots": int(len(ref_imgs)), "episode_iou": float(episode_iou), "hg": analysis,
                                      "p1_diffusion": p1_analysis, "p1_1_counterfactual": counterfactual,
                                      "p1_2_flip_attribution": attribution,
                                      "hg_a1": hg_a1_analysis, "hg_a1_comparison": hg_a1_comparison})
        # save_episode_visualizations(
        #     reference_image=ref_imgs[0],
        #     reference_mask=ref_masks[0],
        #     target_image=tgt_img,
        #     target_gt_mask=tgt_mask,
        #     predicted_mask=pred_mask,
        #     output_dir=args.output_dir,
        #     stem=f"{idx:06d}",
        #     iou=episode_iou,
        # )

        if (idx + 1) % 50 == 0:
            miou = meter.compute_iou()[0]
            pbar.set_description(f'mIoU: {miou:.1f}')
            print(' ')

    # ──────── Final results ────────
    miou = meter.compute_iou()[0].item()
    miou_hg_a1_baseline = meter_hg_a1_baseline.compute_iou()[0].item()
    miou_before = meter_before.compute_iou()[0].item()
    miou_p1_2 = meter_p1_2.compute_iou()[0].item()
    out_str = (f'A1 same-run original Part4 mIoU = {miou_hg_a1_baseline:.3f}; '
               f'active HyperGraph Part4 mIoU = {miou:.3f}; '
               f'delta = {miou - miou_hg_a1_baseline:+.3f}')
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    def avg(key):
        xs = [r["hg"][key] for r in component_records]
        return float(np.mean(xs)) if xs else None
    def correlation(key):
        x=np.array([r["episode_iou"] for r in component_records]); y=np.array([r["hg"][key] for r in component_records])
        return float(np.corrcoef(x,y)[0,1]) if len(x)>=2 and np.isfinite(x).all() and np.isfinite(y).all() and x.std()>1e-12 and y.std()>1e-12 else None
    p1_records = [r["p1_diffusion"] for r in component_records if r["p1_diffusion"] is not None]
    def p1_avg(key):
        xs = [r[key] for r in p1_records]
        return float(np.mean(xs)) if xs else None
    def p1_sum(key):
        return int(sum(r[key] for r in p1_records)) if p1_records else 0
    def p1_correlation(key):
        xs = [(r["episode_iou"], r["p1_diffusion"][key]) for r in component_records if r["p1_diffusion"] is not None]
        if len(xs) < 2:
            return None
        x, y = np.asarray(xs, dtype=float).T
        return float(np.corrcoef(x, y)[0, 1]) if np.isfinite(x).all() and np.isfinite(y).all() and x.std()>1e-12 and y.std()>1e-12 else None
    cf_records = [r["p1_1_counterfactual"] for r in component_records]
    def cf_avg(key):
        return float(np.mean([r[key] for r in cf_records])) if cf_records else None
    def cf_percentile(percentile):
        return float(np.percentile([r["delta_iou"] for r in cf_records], percentile)) if cf_records else None
    def cf_correlation(key, *, absolute_delta=False):
        if len(component_records) < 2:
            return None
        x = np.array([abs(r["p1_1_counterfactual"]["delta_iou"]) if absolute_delta else r["p1_1_counterfactual"]["delta_iou"] for r in component_records])
        y = np.array([r["p1_diffusion"][key] for r in component_records])
        return float(np.corrcoef(x, y)[0, 1]) if np.isfinite(x).all() and np.isfinite(y).all() and x.std()>1e-12 and y.std()>1e-12 else None
    def cf_self_correlation(key):
        if len(cf_records) < 2:
            return None
        x, y = np.asarray([(r["delta_iou"], r[key]) for r in cf_records], dtype=float).T
        return float(np.corrcoef(x, y)[0, 1]) if np.isfinite(x).all() and np.isfinite(y).all() and x.std()>1e-12 and y.std()>1e-12 else None
    summary={"mode":"original_symmetric_hg_gate","num_episodes":len(component_records),"miou":float(miou),"miou_before_same_run":float(miou_before),"miou_delta_same_run":float(miou-miou_before),"hg_gate_mean":avg("hg_gate_mean"),"raw_reliability_mean":avg("raw_reliability_mean"),"negative_retention_ratio_mean":avg("negative_retention_ratio"),"positive_retention_ratio_mean":avg("positive_retention_ratio"),"empty_hypergraph_fallback_count":int(sum(r["hg"]["empty_hypergraph_fallback"] for r in component_records)),"gate_formula_error_abs_max":max([r["hg"]["gate_formula_error_abs_max"] for r in component_records],default=None),"correction_retention_ratio_mean":avg("correction_retention_ratio"),"sign_flip_count_total":int(sum(r["hg"]["sign_flip_count"] for r in component_records)),"corr_episode_iou_vs_negative_retention":correlation("negative_retention_ratio")}
    summary["p1_diffusion_summary"]={"score_change_abs_mean":p1_avg("score_change_abs_mean"),"confidence_mean":p1_avg("confidence_mean"),"conductance_mean":p1_avg("conductance_mean"),"sigma_feat_mean":p1_avg("sigma_feat"),"sigma_rgb_mean":p1_avg("sigma_rgb"),"threshold_flip_fraction_mean":p1_avg("threshold_flip_fraction"),"corr_episode_iou_vs_score_change_abs_mean":p1_correlation("score_change_abs_mean"),"corr_episode_iou_vs_threshold_flip_fraction":p1_correlation("threshold_flip_fraction"),"corr_episode_iou_vs_low_conf_fraction":p1_correlation("low_conf_fraction")}
    summary["p1_1_summary"]={"delta_iou_mean":cf_avg("delta_iou"),"delta_iou_median":cf_percentile(50),"improved_episode_fraction":float(np.mean([r["delta_iou"] > 1e-8 for r in cf_records])) if cf_records else None,"declined_episode_fraction":float(np.mean([r["delta_iou"] < -1e-8 for r in cf_records])) if cf_records else None,"unchanged_episode_fraction":float(np.mean([abs(r["delta_iou"]) <= 1e-8 for r in cf_records])) if cf_records else None,"delta_iou_p01":cf_percentile(1),"delta_iou_p05":cf_percentile(5),"delta_iou_p25":cf_percentile(25),"delta_iou_p50":cf_percentile(50),"delta_iou_p75":cf_percentile(75),"delta_iou_p95":cf_percentile(95),"delta_iou_p99":cf_percentile(99),"delta_iou_min":cf_percentile(0),"delta_iou_max":cf_percentile(100),"corrective_flip_fraction_mean":cf_avg("corrective_flip_fraction"),"harmful_flip_fraction_mean":cf_avg("harmful_flip_fraction"),"corr_delta_iou_vs_patch_flip_fraction":cf_correlation("patch_threshold_flip_fraction"),"corr_abs_delta_iou_vs_patch_flip_fraction":cf_correlation("patch_threshold_flip_fraction",absolute_delta=True),"corr_delta_iou_vs_low_conf_fraction":cf_correlation("normalized_low_conf_fraction"),"corr_delta_iou_vs_raw_score_range":cf_correlation("raw_score_range")}
    total_corrective_fg_to_bg = int(sum(r["corrective_fg_to_bg"] for r in cf_records))
    total_harmful_fg_to_bg = int(sum(r["harmful_fg_to_bg"] for r in cf_records))
    total_corrective_bg_to_fg = int(sum(r["corrective_bg_to_fg"] for r in cf_records))
    total_harmful_bg_to_fg = int(sum(r["harmful_bg_to_fg"] for r in cf_records))
    summary["p1_2_summary"]={"mode":"p1_2_one_sided_dissipative","miou_before_same_run":float(miou_before),"miou_after":float(miou),"miou_delta_same_run":float(miou-miou_before),"delta_iou_mean":cf_avg("delta_iou"),"delta_iou_median":cf_percentile(50),"improved_episode_fraction":float(np.mean([r["delta_iou"] > 1e-8 for r in cf_records])) if cf_records else None,"declined_episode_fraction":float(np.mean([r["delta_iou"] < -1e-8 for r in cf_records])) if cf_records else None,"unchanged_episode_fraction":float(np.mean([abs(r["delta_iou"]) <= 1e-8 for r in cf_records])) if cf_records else None,"upward_candidate_fraction_mean":p1_avg("upward_candidate_fraction"),"downward_candidate_fraction_mean":p1_avg("downward_candidate_fraction"),"rejected_upward_abs_mean":p1_avg("rejected_upward_abs_mean"),"accepted_dissipation_mean":p1_avg("accepted_dissipation_mean"),"patch_bg_to_fg_count_total":p1_sum("patch_bg_to_fg_count"),"patch_fg_to_bg_count_total":p1_sum("patch_fg_to_bg_count"),"final_bg_to_fg_count_total":int(sum(r["bg_to_fg_count"] for r in cf_records)),"final_fg_to_bg_count_total":int(sum(r["fg_to_bg_count"] for r in cf_records)),"fg_to_bg_corrective_fraction_pooled":float(total_corrective_fg_to_bg / max(1, total_corrective_fg_to_bg + total_harmful_fg_to_bg)),"bg_to_fg_corrective_fraction_pooled":float(total_corrective_bg_to_fg / max(1, total_corrective_bg_to_fg + total_harmful_bg_to_fg)),"corr_delta_iou_vs_accepted_dissipation":cf_correlation("accepted_dissipation_mean"),"corr_delta_iou_vs_fg_to_bg_flip_fraction":cf_self_correlation("fg_to_bg_fraction")}
    topology_records = [r["p1_2_flip_attribution"] for r in component_records if r["p1_2_flip_attribution"] is not None]
    neighbor_count_attribution = {str(k): {"p1_2_removed_count": int(sum(r["neighbor_count_attribution"][str(k)]["p1_2_removed_count"] for r in topology_records)), "p1_3_prevented_count": int(sum(r["neighbor_count_attribution"][str(k)]["p1_3_prevented_count"] for r in topology_records))} for k in range(5)}
    for k, row in neighbor_count_attribution.items():
        total_fg = sum(r["neighbor_count_attribution"][k]["gt_fg_fraction"] * r["neighbor_count_attribution"][k]["p1_2_removed_count"] for r in topology_records if r["neighbor_count_attribution"][k]["gt_fg_fraction"] is not None)
        row["gt_fg_fraction"] = float(total_fg / max(1, row["p1_2_removed_count"]))
        row["gt_bg_fraction"] = 1.0 - row["gt_fg_fraction"]
    summary["p1_3_summary"]={"mode":"p1_3_core_preserving_dissipation","miou_before_same_run":float(miou_before),"miou_p1_2_same_run":float(miou_p1_2),"miou_p1_3":float(miou),"delta_p1_2_vs_before":float(miou_p1_2-miou_before),"delta_p1_3_vs_before":float(miou-miou_before),"delta_p1_3_vs_p1_2":float(miou-miou_p1_2),"episode_delta_p1_3_vs_p1_2_mean":cf_avg("delta_p1_3_vs_p1_2"),"episode_delta_p1_3_vs_p1_2_median":float(np.median([r["delta_p1_3_vs_p1_2"] for r in cf_records])) if cf_records else None,"p1_2_patch_fg_to_bg_total":int(sum(r["p1_2_patch_fg_to_bg_count"] for r in topology_records)),"p1_3_patch_fg_to_bg_total":int(sum(r["p1_3_patch_fg_to_bg_count"] for r in topology_records)),"prevented_patch_fg_to_bg_total":int(sum(r["prevented_patch_fg_to_bg_count"] for r in topology_records)),"p1_2_final_fg_to_bg_total":int(sum(r["p1_2_final_fg_to_bg_count"] for r in cf_records)),"p1_3_final_fg_to_bg_total":int(sum(r["p1_3_final_fg_to_bg_count"] for r in cf_records)),"prevented_final_fg_to_bg_total":int(sum(r["prevented_final_fg_to_bg_count"] for r in cf_records)),"protected_true_fg_total":int(sum(r["protected_true_fg_count"] for r in cf_records)),"protected_false_fg_total":int(sum(r["protected_false_fg_count"] for r in cf_records)),"protected_true_fg_fraction":float(sum(r["protected_true_fg_count"] for r in cf_records) / max(1, sum(r["protected_true_fg_count"] + r["protected_false_fg_count"] for r in cf_records))),"core_protection_mean_fg":p1_avg("core_protection_mean_fg"),"dissipation_reduction_mean":p1_avg("dissipation_reduction_mean"),"monotonicity_violation_count_total":p1_sum("monotonicity_violation_count"),"p1_2_lower_bound_violation_count_total":p1_sum("p1_2_lower_bound_violation_count"),"neighbor_count_attribution":neighbor_count_attribution}
    def rank_auc(corrective, harmful):
        if corrective.size == 0 or harmful.size == 0:
            return None
        values = np.concatenate([corrective, harmful])
        labels = np.concatenate([np.zeros(corrective.size, dtype=bool), np.ones(harmful.size, dtype=bool)])
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end + 1) / 2.0
            start = end
        n_h, n_c = harmful.size, corrective.size
        return float((ranks[labels].sum() - n_h * (n_h + 1) / 2.0) / (n_h * n_c))
    attribution_signals, best_signal, best_auc = {}, None, None
    for key, data in pooled_signals.items():
        corrective = np.concatenate(data["corrective"]) if data["corrective"] else np.empty(0)
        harmful = np.concatenate(data["harmful"]) if data["harmful"] else np.empty(0)
        auc = rank_auc(corrective, harmful)
        pooled_std = np.sqrt((corrective.var() + harmful.var()) / 2.0) if corrective.size > 1 and harmful.size > 1 else None
        diffs = np.asarray(data["episode_diffs"], dtype=float)
        attribution_signals[key] = {"auc_harmful_positive": auc, "corrective_mean": float(corrective.mean()) if corrective.size else None, "harmful_mean": float(harmful.mean()) if harmful.size else None, "mean_diff_corrective_minus_harmful": float(corrective.mean() - harmful.mean()) if corrective.size and harmful.size else None, "cohens_d": float((corrective.mean() - harmful.mean()) / (pooled_std + 1e-8)) if pooled_std is not None else None, "median_diff_harmful_minus_corrective": float(np.median(harmful) - np.median(corrective)) if corrective.size and harmful.size else None, "episode_diff_mean": float(diffs.mean()) if diffs.size else None, "episode_diff_median": float(np.median(diffs)) if diffs.size else None, "episode_diff_positive_fraction": float((diffs > 0).mean()) if diffs.size else None}
        if auc is not None and (best_auc is None or auc > best_auc):
            best_signal, best_auc = key, auc
    def episode_card(record):
        attr, cf = record["p1_2_flip_attribution"], record["p1_1_counterfactual"]
        stats = attr["signals"]
        return {"episode_index": record["episode_index"], "class_id": record["class_id"], "iou_before": cf["iou_before"], "iou_after": cf["iou_after"], "delta_iou": cf["delta_iou"], "removed_patch_count": attr["removed_patch_count"], "corrective_removed_patch_count": attr["corrective_removed_patch_count"], "harmful_removed_patch_count": attr["harmful_removed_patch_count"], "final_corrective_fg_to_bg": cf["corrective_fg_to_bg"], "final_harmful_fg_to_bg": cf["harmful_fg_to_bg"], "sf_harmful_mean": stats["sf"]["harmful"]["mean"], "sbn_harmful_mean": stats["sbn"]["harmful"]["mean"], "cand_harmful_mean": stats["cand_soft"]["harmful"]["mean"], "seed_harmful_mean": stats["seed_prior"]["harmful"]["mean"], "semantic_margin_harmful_mean": stats["semantic_margin"]["harmful"]["mean"], "fg_support_harmful_mean": stats["fg_support_mean"]["harmful"]["mean"], "dissipation_gap_harmful_mean": stats["dissipation_gap"]["harmful"]["mean"]}
    ranked_records = [r for r in component_records if r["p1_2_flip_attribution"] is not None]
    summary["p1_2_flip_attribution_summary"]={"num_episodes":len(ranked_records),"positive_class":"harmful_removed_patch","auc_interpretation":"AUC > 0.5 means higher signal values are more associated with a harmful removal.","pooled_corrective_removed_patch_count":int(sum(r["p1_2_flip_attribution"]["corrective_removed_patch_count"] for r in ranked_records)),"pooled_harmful_removed_patch_count":int(sum(r["p1_2_flip_attribution"]["harmful_removed_patch_count"] for r in ranked_records)),"pooled_corrective_final_fg_to_bg":total_corrective_fg_to_bg,"pooled_harmful_final_fg_to_bg":total_harmful_fg_to_bg,"signals":attribution_signals,"best_auc_signal":best_signal,"best_auc_value":best_auc,"top_harmful_episodes":[episode_card(r) for r in sorted(ranked_records,key=lambda r:r["p1_1_counterfactual"]["delta_iou"])[:20]],"top_beneficial_episodes":[episode_card(r) for r in sorted(ranked_records,key=lambda r:r["p1_1_counterfactual"]["delta_iou"],reverse=True)[:20]]}
    hg_a1_records = [r for r in component_records if r["hg_a1"] is not None]
    hg_a1_deltas = np.asarray([r["hg_a1_comparison"]["delta_iou"] for r in hg_a1_records], dtype=float)
    summary["hg_a1_summary"] = {
        "mode": "a1_sparse_signed_hypergraph_consolidation",
        "baseline_mode": "original_symmetric_hg_gate",
        "miou_baseline_same_run": float(miou_hg_a1_baseline),
        "miou_active": float(miou),
        "miou_delta_same_run": float(miou - miou_hg_a1_baseline),
        "episode_delta_iou_mean": float(hg_a1_deltas.mean()) if hg_a1_deltas.size else None,
        "episode_delta_iou_median": float(np.median(hg_a1_deltas)) if hg_a1_deltas.size else None,
        "improved_episode_fraction": float((hg_a1_deltas > 0).mean()) if hg_a1_deltas.size else None,
        "declined_episode_fraction": float((hg_a1_deltas < 0).mean()) if hg_a1_deltas.size else None,
        "num_hyperedges_mean": float(np.mean([r["hg_a1"]["num_hyperedges"] for r in hg_a1_records])) if hg_a1_records else None,
        "isolated_node_fraction_mean": float(np.mean([r["hg_a1"]["isolated_node_fraction"] for r in hg_a1_records])) if hg_a1_records else None,
        "fg_to_bg_total": int(sum(r["hg_a1_comparison"]["fg_to_bg_count"] for r in hg_a1_records)),
        "bg_to_fg_total": int(sum(r["hg_a1_comparison"]["bg_to_fg_count"] for r in hg_a1_records)),
        "corrective_flip_total": int(sum(r["hg_a1_comparison"]["corrective_flip_count"] for r in hg_a1_records)),
        "harmful_flip_total": int(sum(r["hg_a1_comparison"]["harmful_flip_count"] for r in hg_a1_records)),
    }
    payload={"component":{"name":"part4_evidence_consistency_hypergraph_gate","mode":"original_symmetric_hg_gate","description":"Part-4 cluster corrections are symmetrically attenuated by an evidence-consistency hypergraph gate.","gate_formula":"gate = 0.5 + 0.5 * raw_reliability","hyperedges":["sf_foreground_support","candidate_support","seed_prior_support"],"gate_range":[.5,1.]},"run":{"dataset":str(args.dataset),"exp_name":str(args.exp_name),"seed":int(args.seed),"num_episodes":len(component_records),"output_dir":str(args.output_dir)},"summary":summary,"episodes":component_records}
    payload["baseline_component"] = payload["component"]
    payload["component"] = {"name": "part4_sparse_signed_hypergraph_consolidation", "mode": "a1_sparse_signed_hypergraph_consolidation", "description": "Sparse cluster-level FG, candidate, seed and BG-conflict groups produce a signed correction. The same-run baseline uses the original symmetric Part4 gate."}
    payload["summary"] = summary["hg_a1_summary"]
    payload["episodes"] = [
        {
            "episode_index": r["episode_index"], "class_id": r["class_id"],
            "episode_iou_active": r["episode_iou"],
            "hg_baseline": r["hg"], "hg_a1": r["hg_a1"],
            "hg_a1_comparison": r["hg_a1_comparison"],
        }
        for r in component_records
    ]
    analysis_path=join(args.output_dir,"hg_a1_component_analysis.json")
    with open(analysis_path,"w",encoding="utf-8") as fp: json.dump(payload,fp,indent=2,ensure_ascii=False)
    print(f"HG component analysis saved to: {analysis_path}")
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FoRIS inference', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
