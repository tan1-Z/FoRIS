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

        fg_union = area_union[1].clamp_min(1.0)
        episode_iou = (area_inter[1] / fg_union * 100.0).item()
        analysis = getattr(model, "last_hg_part4_analysis", None)
        if analysis is not None:
            component_records.append({"episode_index": int(idx), "class_id": int(class_id[0].item()),
                                      "num_shots": int(len(ref_imgs)), "episode_iou": float(episode_iou), "hg": analysis})
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
    def avg(key):
        xs = [r["hg"][key] for r in component_records]
        return float(np.mean(xs)) if xs else None
    def correlation(key):
        x=np.array([r["episode_iou"] for r in component_records]); y=np.array([r["hg"][key] for r in component_records])
        return float(np.corrcoef(x,y)[0,1]) if len(x)>=2 and np.isfinite(x).all() and np.isfinite(y).all() and x.std()>1e-12 and y.std()>1e-12 else None
    summary={"mode":"original_symmetric_hg_gate","num_episodes":len(component_records),"miou":float(miou),"hg_gate_mean":avg("hg_gate_mean"),"raw_reliability_mean":avg("raw_reliability_mean"),"negative_retention_ratio_mean":avg("negative_retention_ratio"),"positive_retention_ratio_mean":avg("positive_retention_ratio"),"empty_hypergraph_fallback_count":int(sum(r["hg"]["empty_hypergraph_fallback"] for r in component_records)),"gate_formula_error_abs_max":max([r["hg"]["gate_formula_error_abs_max"] for r in component_records],default=None),"correction_retention_ratio_mean":avg("correction_retention_ratio"),"sign_flip_count_total":int(sum(r["hg"]["sign_flip_count"] for r in component_records)),"corr_episode_iou_vs_negative_retention":correlation("negative_retention_ratio")}
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
