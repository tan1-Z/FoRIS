"""INSID3 inference script."""

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
    meter = AverageMeter(args.dataset, ds.class_ids)
    evaluation_records = []
    counterfactual_records = []
    hypergraph_tv_records = []

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
        meter.update(area_inter, area_union, batch['class_id'].cuda())

        fg_union = area_union[1].clamp_min(1.0)
        episode_iou = (area_inter[1] / fg_union * 100.0).item()
        evaluation_records.append({
            "episode_index": int(idx),
            "class_id": int(torch.as_tensor(batch["class_id"]).reshape(-1)[0]),
            "episode_iou": float(episode_iou),
        })
        analysis = getattr(model, "last_reference_counterfactual_analysis", None)
        if analysis is not None:
            counterfactual_records.append({
                "episode_index": int(idx),
                "class_id": int(torch.as_tensor(batch["class_id"]).reshape(-1)[0]),
                **analysis,
            })
        tv_analysis = getattr(model, "last_hypergraph_tv_analysis", None)
        if tv_analysis is not None:
            hypergraph_tv_records.append({
                "episode_index": int(idx),
                "class_id": int(torch.as_tensor(batch["class_id"]).reshape(-1)[0]),
                **tv_analysis,
            })
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
    out_str = f'mIoU = {miou:.1f}'
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    evaluation_summary = {
        "artifact_type": "final_model_evaluation",
        "metric_semantics": "Metrics are computed from the single final prediction emitted by this run.",
        "miou": float(miou),
        "num_episodes": len(evaluation_records),
        "reference_counterfactual_view": bool(args.reference_counterfactual_view),
        "reference_counterfactual_blend": float(args.reference_counterfactual_blend),
        "hypergraph_tv": bool(args.hypergraph_tv),
    }
    evaluation_path = join(args.output_dir, "evaluation.json")
    with open(evaluation_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"summary": evaluation_summary, "episodes": evaluation_records},
            fp, indent=2, ensure_ascii=False,
        )
    print(f"Final evaluation saved to: {evaluation_path}")
    numeric_keys = (
        "num_fg_tokens", "num_fg_prototypes", "mu_original_view_cosine",
        "prototype_original_view_cosine_mean", "mu_original_fused_cosine",
    )
    summary = {
        "artifact_type": "module_diagnostics",
        "metric_semantics": "This file does not contain a separate model evaluation; see evaluation.json.",
        "mode": "reference_foreground_counterfactual_view",
        "enabled": bool(args.reference_counterfactual_view),
        "blend": float(args.reference_counterfactual_blend),
        "num_episodes": len(counterfactual_records),
        "fallback_count": int(sum(record.get("fallback", False) for record in counterfactual_records)),
    }
    for key in numeric_keys:
        values = [record[key] for record in counterfactual_records if key in record]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
    tv_numeric_keys = (
        "num_hyperedges", "num_local_hyperedges",
        "num_fg_anchor_hyperedges", "num_bg_anchor_hyperedges",
        "mean_hyperedge_weight", "mean_node_degree", "max_node_degree",
        "iterations", "primal_residual", "dual_residual",
        "mean_absolute_score_change",
    )
    tv_summary = {
        "artifact_type": "module_diagnostics",
        "metric_semantics": "This file does not contain a separate model evaluation; see evaluation.json.",
        "mode": "post_part4_hypergraph_tv",
        "enabled": bool(args.hypergraph_tv),
        "lambda": float(args.hypergraph_tv_lambda),
        "max_iterations": int(args.hypergraph_tv_iterations),
        "local_similarity": float(args.hypergraph_tv_local_similarity),
        "anchor_ratio": float(args.hypergraph_tv_anchor_ratio),
        "primal_step": float(args.hypergraph_tv_primal_step),
        "dual_step": float(args.hypergraph_tv_dual_step),
        "tolerance": float(args.hypergraph_tv_tolerance),
        "num_episodes": len(hypergraph_tv_records),
        "fallback_count": int(sum(
            record.get("fallback", False) for record in hypergraph_tv_records
        )),
    }
    for key in tv_numeric_keys:
        values = [
            record[key] for record in hypergraph_tv_records
            if record.get(key) is not None
        ]
        tv_summary[f"{key}_mean"] = float(np.mean(values)) if values else None
    diagnostics_path = join(args.output_dir, "module_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "artifact_type": "module_diagnostics",
                "metric_semantics": "Module diagnostics only; final metrics are in evaluation.json.",
                "reference_counterfactual": {
                    "summary": summary,
                    "episodes": counterfactual_records,
                },
                "hypergraph_tv": {
                    "summary": tv_summary,
                    "episodes": hypergraph_tv_records,
                },
            },
            fp, indent=2, ensure_ascii=False,
        )
    print(f"Module diagnostics saved to: {diagnostics_path}")
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INSID3 inference', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
