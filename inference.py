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
    meter_before = AverageMeter(args.dataset, list(ds.class_ids))
    component_records = []

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
        ).squeeze(0).squeeze(0) > 0.5

        tgt_ignore_idx = batch.get('tgt_ignore_idx')
        if tgt_ignore_idx is not None:
            tgt_ignore_idx = F.interpolate(
                tgt_ignore_idx.unsqueeze(0).unsqueeze(0).float(),
                size=pred_hw, mode='nearest',
            ).squeeze(0).squeeze(0) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=tgt_ignore_idx,
        )
        class_id = torch.as_tensor(batch['class_id'], device=args.device).reshape(-1)
        meter.update(area_inter, area_union, class_id)

        before_mask = getattr(model, "last_p1_before_mask", None)
        if before_mask is None:
            raise RuntimeError("P1.1 counterfactual mask was not produced.")
        before_inter, before_union = Evaluator.classify_prediction(
            before_mask, tgt_mask, tgt_ignore_idx=tgt_ignore_idx,
        )
        meter_before.update(before_inter, before_union, class_id)

        fg_union = area_union[1].clamp_min(1.0)
        episode_iou = (area_inter[1] / fg_union * 100.0).item()
        before_iou = (before_inter[1] / before_union[1].clamp_min(1.0) * 100.0).item()
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
        counterfactual = {"iou_before": float(before_iou), "iou_after": float(episode_iou), "delta_iou": float(episode_iou - before_iou), "flip_count": flip_count, "flip_fraction": float(flip_count / max(1, valid_count)), "fg_to_bg_count": int(fg_to_bg.sum()), "bg_to_fg_count": int(bg_to_fg.sum()), "corrective_flip_count": corrective_flip_count, "harmful_flip_count": harmful_flip_count, "corrective_flip_fraction": float(corrective_flip_count / max(1, flip_count)), "harmful_flip_fraction": float(harmful_flip_count / max(1, flip_count)), "net_corrective_flips": corrective_flip_count - harmful_flip_count, "corrective_bg_to_fg": corrective_bg_to_fg, "harmful_bg_to_fg": harmful_bg_to_fg, "corrective_fg_to_bg": corrective_fg_to_bg, "harmful_fg_to_bg": harmful_fg_to_bg}
        analysis = getattr(model, "last_hg_part4_analysis", None)
        p1_analysis = getattr(model, "last_p1_diffusion_analysis", None)
        if analysis is not None:
            component_records.append({"episode_index": int(idx), "class_id": int(class_id[0].item()),
                                      "num_shots": int(len(ref_imgs)), "episode_iou": float(episode_iou), "hg": analysis,
                                      "p1_diffusion": p1_analysis, "p1_1_counterfactual": counterfactual})
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
    miou_before = meter_before.compute_iou()[0].item()
    out_str = f'mIoU after P1.1 = {miou:.1f}; same-run before P1.1 = {miou_before:.1f}; delta = {miou - miou_before:.3f}'
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
    summary={"mode":"original_symmetric_hg_gate","num_episodes":len(component_records),"miou":float(miou),"miou_before_same_run":float(miou_before),"miou_delta_same_run":float(miou-miou_before),"hg_gate_mean":avg("hg_gate_mean"),"raw_reliability_mean":avg("raw_reliability_mean"),"negative_retention_ratio_mean":avg("negative_retention_ratio"),"positive_retention_ratio_mean":avg("positive_retention_ratio"),"empty_hypergraph_fallback_count":int(sum(r["hg"]["empty_hypergraph_fallback"] for r in component_records)),"gate_formula_error_abs_max":max([r["hg"]["gate_formula_error_abs_max"] for r in component_records],default=None),"correction_retention_ratio_mean":avg("correction_retention_ratio"),"sign_flip_count_total":int(sum(r["hg"]["sign_flip_count"] for r in component_records)),"corr_episode_iou_vs_negative_retention":correlation("negative_retention_ratio")}
    summary["p1_diffusion_summary"]={"score_change_abs_mean":p1_avg("score_change_abs_mean"),"confidence_mean":p1_avg("confidence_mean"),"conductance_mean":p1_avg("conductance_mean"),"sigma_feat_mean":p1_avg("sigma_feat"),"sigma_rgb_mean":p1_avg("sigma_rgb"),"threshold_flip_fraction_mean":p1_avg("threshold_flip_fraction"),"corr_episode_iou_vs_score_change_abs_mean":p1_correlation("score_change_abs_mean"),"corr_episode_iou_vs_threshold_flip_fraction":p1_correlation("threshold_flip_fraction"),"corr_episode_iou_vs_low_conf_fraction":p1_correlation("low_conf_fraction")}
    summary["p1_1_summary"]={"delta_iou_mean":cf_avg("delta_iou"),"delta_iou_median":cf_percentile(50),"improved_episode_fraction":float(np.mean([r["delta_iou"] > 1e-8 for r in cf_records])) if cf_records else None,"declined_episode_fraction":float(np.mean([r["delta_iou"] < -1e-8 for r in cf_records])) if cf_records else None,"unchanged_episode_fraction":float(np.mean([abs(r["delta_iou"]) <= 1e-8 for r in cf_records])) if cf_records else None,"delta_iou_p01":cf_percentile(1),"delta_iou_p05":cf_percentile(5),"delta_iou_p25":cf_percentile(25),"delta_iou_p50":cf_percentile(50),"delta_iou_p75":cf_percentile(75),"delta_iou_p95":cf_percentile(95),"delta_iou_p99":cf_percentile(99),"delta_iou_min":cf_percentile(0),"delta_iou_max":cf_percentile(100),"corrective_flip_fraction_mean":cf_avg("corrective_flip_fraction"),"harmful_flip_fraction_mean":cf_avg("harmful_flip_fraction"),"corr_delta_iou_vs_patch_flip_fraction":cf_correlation("patch_threshold_flip_fraction"),"corr_abs_delta_iou_vs_patch_flip_fraction":cf_correlation("patch_threshold_flip_fraction",absolute_delta=True),"corr_delta_iou_vs_low_conf_fraction":cf_correlation("normalized_low_conf_fraction"),"corr_delta_iou_vs_raw_score_range":cf_correlation("raw_score_range")}
    payload={"component":{"name":"part4_evidence_consistency_hypergraph_gate","mode":"original_symmetric_hg_gate","description":"Part-4 cluster corrections are symmetrically attenuated by an evidence-consistency hypergraph gate.","gate_formula":"gate = 0.5 + 0.5 * raw_reliability","hyperedges":["sf_foreground_support","candidate_support","seed_prior_support"],"gate_range":[.5,1.]},"run":{"dataset":str(args.dataset),"exp_name":str(args.exp_name),"seed":int(args.seed),"num_episodes":len(component_records),"output_dir":str(args.output_dir)},"summary":summary,"episodes":component_records}
    analysis_path=join(args.output_dir,"hg_part4_component_analysis.json")
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
