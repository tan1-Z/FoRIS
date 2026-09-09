"""Evaluate FoRIS with Conservative Risk-Gated Affinity Fusion."""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_foris_from_args
from utils.metrics import AverageMeter, Evaluator


def _first_item_collate(items):
    """Keep PIL images and variable-length reference lists unbatched."""
    return items[0]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return (
        F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=size,
            mode="nearest",
        )[0, 0]
        > 0.5
    )


def evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    output_dir: Path,
) -> float:
    device = torch.device(args.device)
    dataset = build_dataset(args.dataset, args=args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_first_item_collate,
        pin_memory=device.type == "cuda",
    )
    meter = AverageMeter(args.dataset, list(dataset.class_ids), device=device)

    routing_file = None
    if args.save_routing:
        routing_file = (output_dir / "routing.jsonl").open("w", encoding="utf-8")

    episode_times: list[float] = []
    hard_routes = 0
    selected_layer_counts: dict[int, int] = {}

    try:
        progress = tqdm(loader, ncols=100)
        for episode_idx, batch in enumerate(progress):
            model._ref_images = None
            model._ref_masks = None
            for ref_image, ref_mask in zip(batch["ref_imgs"], batch["ref_masks"]):
                model.set_reference(ref_image, ref_mask)
            model.set_target(batch["tgt_img"])

            _synchronize(device)
            start = time.perf_counter()
            prediction = model.segment()
            _synchronize(device)
            episode_times.append((time.perf_counter() - start) * 1000.0)

            pred_2d = prediction.detach()
            while pred_2d.ndim > 2:
                pred_2d = pred_2d.squeeze(0)
            target_mask = _prepare_mask(batch["tgt_mask"], tuple(pred_2d.shape))

            ignore_mask = batch.get("tgt_ignore_idx")
            if ignore_mask is not None:
                ignore_mask = _prepare_mask(ignore_mask, tuple(pred_2d.shape))

            intersection, union = Evaluator.classify_prediction(
                pred_2d,
                target_mask,
                tgt_ignore_idx=ignore_mask,
            )
            class_id = torch.as_tensor(batch["class_id"], device=device).reshape(-1)
            meter.update(intersection, union, class_id)

            routing = dict(model.last_routing_info)
            best_layer = int(routing.get("best_layer", -1))
            selected_layer_counts[best_layer] = selected_layer_counts.get(best_layer, 0) + 1
            hard_routes += int(bool(routing.get("hard_routed", False)))

            fg_union = union[1].clamp_min(1.0)
            episode_iou = float((intersection[1] / fg_union * 100.0).item())
            if routing_file is not None:
                routing_file.write(
                    json.dumps(
                        {
                            "episode": episode_idx,
                            "class_id": int(class_id.item()),
                            "iou": episode_iou,
                            "runtime_ms": episode_times[-1],
                            **routing,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (episode_idx + 1) % 20 == 0:
                miou = float(meter.compute_iou()[0].item())
                progress.set_description(
                    f"mIoU {miou:.2f} | layer {best_layer} | "
                    f"{episode_times[-1]:.0f} ms"
                )
    finally:
        if routing_file is not None:
            routing_file.close()

    miou = float(meter.compute_iou()[0].item())
    runtime = np.asarray(episode_times, dtype=np.float64)
    summary = {
        "miou": miou,
        "episodes": len(episode_times),
        "runtime_ms_mean": float(runtime.mean()) if runtime.size else None,
        "runtime_ms_p50": float(np.percentile(runtime, 50)) if runtime.size else None,
        "runtime_ms_p95": float(np.percentile(runtime, 95)) if runtime.size else None,
        "hard_route_fraction": hard_routes / max(1, len(episode_times)),
        "selected_layer_counts": {
            str(layer): count for layer, count in sorted(selected_layer_counts.items())
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return miou


def main(args: argparse.Namespace) -> float:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    model = build_foris_from_args(args).eval()
    return evaluate(args, model, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "FoRIS CRGAF segmentation inference",
        parents=[opts.get_args_parser()],
    )
    main(parser.parse_args())
