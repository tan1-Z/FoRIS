"""Training-free hypergraph total-variation refinement for FoRIS scores."""

from __future__ import annotations

import torch
import torch.nn.functional as F



def _local_neighborhood_edges(
    features: torch.Tensor,
    *,
    similarity_threshold: float,
) -> list[tuple[list[int], float]]:
    """Create 3-5 node local hyperedges from a target feature grid."""
    channels, height, width = features.shape
    tokens = F.normalize(features.permute(1, 2, 0).reshape(-1, channels), p=2, dim=1)
    edges: list[tuple[list[int], float]] = []
    for row in range(height):
        for col in range(width):
            center = row * width + col
            neighbors: list[int] = []
            for rr in range(max(0, row - 1), min(height, row + 2)):
                for cc in range(max(0, col - 1), min(width, col + 2)):
                    if rr != row or cc != col:
                        neighbors.append(rr * width + cc)
            if len(neighbors) < 2:
                continue
            neighbor_tensor = torch.tensor(neighbors, device=features.device)
            similarity = tokens[neighbor_tensor] @ tokens[center]
            valid = neighbor_tensor[similarity >= similarity_threshold]
            if valid.numel() < 2:
                continue
            top_count = min(4, int(valid.numel()))
            selected_similarity, selected_order = similarity[similarity >= similarity_threshold].topk(top_count)
            selected = valid[selected_order]
            nodes = [center, *selected.tolist()]
            edges.append((nodes, float(selected_similarity.mean().item())))
    return edges


def _anchor_edges(
    confidence: torch.Tensor,
    local_edges: list[tuple[list[int], float]],
    *,
    anchor_value: float,
    anchor_ratio: float,
    view_reliability: torch.Tensor,
) -> list[tuple[list[int], float, float]]:
    """Attach a fixed FG/BG anchor to locally coherent high-confidence groups."""
    flat_confidence = confidence.reshape(-1)
    num_anchors = max(1, int(round(flat_confidence.numel() * anchor_ratio)))
    anchor_centers = set(flat_confidence.topk(num_anchors).indices.tolist())
    result: list[tuple[list[int], float, float]] = []
    flat_reliability = view_reliability.reshape(-1)
    for nodes, local_similarity in local_edges:
        center = nodes[0]
        if center not in anchor_centers:
            continue
        reliability = float(flat_reliability[center].item())
        weight = max(0.0, float(flat_confidence[center].item()))
        weight *= max(0.0, local_similarity) * reliability
        if weight > 0.0:
            result.append((nodes, anchor_value, weight))
    return result


def _pack_hyperedges(
    local_edges: list[tuple[list[int], float]],
    fg_anchor_edges: list[tuple[list[int], float, float]],
    bg_anchor_edges: list[tuple[list[int], float, float]],
    *,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack target nodes and optional fixed anchors into fixed-size edge tensors."""
    records: list[tuple[list[int], float | None, float]] = []
    records.extend((nodes, None, weight) for nodes, weight in local_edges)
    records.extend((nodes, anchor, weight) for nodes, anchor, weight in fg_anchor_edges)
    records.extend((nodes, anchor, weight) for nodes, anchor, weight in bg_anchor_edges)
    if not records:
        raise RuntimeError("No valid hyperedges were constructed")

    max_nodes = max(len(nodes) + int(anchor is not None) for nodes, anchor, _ in records)
    edge_nodes = torch.full(
        (len(records), max_nodes), -2, device=device, dtype=torch.long
    )
    fixed_values = torch.zeros(
        (len(records), max_nodes), device=device, dtype=dtype
    )
    raw_weights = torch.empty(len(records), device=device, dtype=dtype)
    for edge, (nodes, anchor, weight) in enumerate(records):
        node_tensor = torch.tensor(nodes, device=device, dtype=torch.long)
        edge_nodes[edge, : node_tensor.numel()] = node_tensor
        if anchor is not None:
            edge_nodes[edge, node_tensor.numel()] = -1
            fixed_values[edge, node_tensor.numel()] = anchor
        raw_weights[edge] = weight

    target_mask = edge_nodes >= 0
    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.scatter_add_(0, edge_nodes[target_mask], torch.ones_like(edge_nodes[target_mask], dtype=dtype))
    edge_degree = torch.ones(len(records), device=device, dtype=dtype)
    for edge in range(len(records)):
        members = edge_nodes[edge][edge_nodes[edge] >= 0]
        edge_degree[edge] = degree[members].mean().clamp_min(1.0)
    weights = raw_weights / edge_degree
    weights = weights / weights.mean().clamp_min(1e-6)
    return edge_nodes, fixed_values, weights, degree


def _project_scaled_simplex(values: torch.Tensor, mass: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Project each padded row onto a non-negative simplex with row mass."""
    original_shape = values.shape
    values = values.reshape(values.shape[0], -1)
    valid = valid.reshape(valid.shape[0], -1)
    masked = values.masked_fill(~valid, float("-inf"))
    sorted_values = masked.sort(dim=1, descending=True).values
    cumsum = sorted_values.cumsum(dim=1)
    ranks = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    threshold = (cumsum - mass.unsqueeze(1)) / ranks.unsqueeze(0)
    support = sorted_values > threshold
    rho = support.sum(dim=1).clamp_min(1) - 1
    theta = threshold.gather(1, rho.unsqueeze(1)).squeeze(1)
    projected = (values - theta.unsqueeze(1)).clamp_min(0.0)
    return projected.masked_fill(~valid, 0.0).reshape(original_shape)


def _evidence_interval(
    evidence_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    max_width: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an uncalibrated, evidence-disagreement interval around each score."""
    evidence = torch.stack([item.clamp(0.0, 1.0) for item in evidence_maps], dim=0)
    median = evidence.median(dim=0).values
    disagreement = (evidence - median).abs().mean(dim=0)
    width = (scale * disagreement).clamp(0.0, max_width)
    return width, disagreement, median


def _interval_fidelity_prox(
    proposal: torch.Tensor,
    s0: torch.Tensor,
    fidelity: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    primal_step: float,
    epsilon: float,
) -> torch.Tensor:
    """Exact separable proximal map for interval fidelity plus an s0 tether."""
    interval_weight = fidelity - epsilon
    middle = (proposal + primal_step * epsilon * s0) / (1.0 + primal_step * epsilon)
    middle = middle.clamp(min=lower, max=upper)
    lower_candidate = (
        proposal + primal_step * (epsilon * s0 + interval_weight * lower)
    ) / (1.0 + primal_step * (epsilon + interval_weight))
    lower_candidate = torch.minimum(lower_candidate, lower)
    upper_candidate = (
        proposal + primal_step * (epsilon * s0 + interval_weight * upper)
    ) / (1.0 + primal_step * (epsilon + interval_weight))
    upper_candidate = torch.maximum(upper_candidate, upper)

    def energy(candidate: torch.Tensor) -> torch.Tensor:
        distance = (candidate - candidate.clamp(min=lower, max=upper)).square()
        return (
            0.5 * (candidate - proposal).square() / primal_step
            + 0.5 * interval_weight * distance
            + 0.5 * epsilon * (candidate - s0).square()
        )

    candidates = torch.stack([lower_candidate, middle, upper_candidate], dim=0)
    energies = torch.stack([energy(item) for item in candidates], dim=0)
    best = energies.argmin(dim=0, keepdim=True)
    return candidates.gather(0, best).squeeze(0).clamp(0.0, 1.0)


def _build_hyperedges_fast(
    target_features: torch.Tensor,
    sf: torch.Tensor,
    sbn: torch.Tensor,
    view_reliability: torch.Tensor,
    *,
    similarity_threshold: float,
    anchor_ratio: float,
    fg_anchor_margin: float,
    bg_anchor_margin: float,
    min_fg_view_reliability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Vectorized construction of local and anchored hyperedges on GPU."""
    channels, height, width = target_features.shape
    num_nodes = height * width
    tokens = F.normalize(
        target_features.permute(1, 2, 0).reshape(num_nodes, channels), p=2, dim=1
    )
    rows = torch.arange(height, device=target_features.device).view(height, 1).expand(height, width).reshape(-1)
    cols = torch.arange(width, device=target_features.device).view(1, width).expand(height, width).reshape(-1)
    center = torch.arange(num_nodes, device=target_features.device)
    offsets = torch.tensor(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        device=target_features.device,
    )
    neighbor_rows = rows[:, None] + offsets[None, :, 0]
    neighbor_cols = cols[:, None] + offsets[None, :, 1]
    valid_neighbor = (
        (neighbor_rows >= 0) & (neighbor_rows < height)
        & (neighbor_cols >= 0) & (neighbor_cols < width)
    )
    safe_rows = neighbor_rows.clamp(0, height - 1)
    safe_cols = neighbor_cols.clamp(0, width - 1)
    neighbor_index = safe_rows * width + safe_cols
    neighbor_similarity = (tokens[neighbor_index] * tokens[:, None]).sum(dim=2)
    eligible = valid_neighbor & (neighbor_similarity >= similarity_threshold)
    ranked_similarity = neighbor_similarity.masked_fill(~eligible, float("-inf"))
    selected_similarity, selected_position = ranked_similarity.topk(4, dim=1)
    selected_index = neighbor_index.gather(1, selected_position)
    selected_valid = torch.isfinite(selected_similarity)
    selected_index = selected_index.masked_fill(~selected_valid, -2)
    selected_count = selected_valid.sum(dim=1)
    keep = selected_count >= 2
    local_nodes = torch.cat([center[:, None], selected_index], dim=1)[keep]
    local_weight = (
        selected_similarity.masked_fill(~selected_valid, 0.0).sum(dim=1)
        / selected_count.clamp_min(1)
    ).clamp_min(0.0)[keep]
    local_centers = center[keep]
    if local_nodes.numel() == 0:
        raise RuntimeError("No local hyperedges passed the similarity threshold")

    flat_sf = sf.reshape(-1)
    flat_sbn = sbn.reshape(-1)
    flat_view = view_reliability.reshape(-1)
    anchors = max(1, int(round(num_nodes * anchor_ratio)))

    def make_anchors(
        margin: torch.Tensor,
        anchor_value: float,
        reliability: torch.Tensor,
        minimum_margin: float,
        minimum_reliability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        margin_at_centers = margin[local_centers]
        reliability_at_centers = reliability[local_centers]
        eligible = (
            (margin_at_centers >= minimum_margin)
            & (reliability_at_centers >= minimum_reliability)
        )
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            empty_nodes = torch.empty((0, 6), device=target_features.device, dtype=torch.long)
            empty_fixed = torch.empty((0, 6), device=target_features.device, dtype=target_features.dtype)
            empty_weight = torch.empty((0,), device=target_features.device, dtype=target_features.dtype)
            return empty_nodes, empty_fixed, empty_weight, eligible_count
        count = min(anchors, eligible_count)
        ranked_margin = margin_at_centers.masked_fill(~eligible, float("-inf"))
        chosen = ranked_margin.topk(count).indices
        nodes = torch.cat([
            local_nodes[chosen],
            torch.full((count, 1), -1, dtype=torch.long, device=target_features.device),
        ], dim=1)
        fixed = torch.zeros((count, 6), device=target_features.device, dtype=target_features.dtype)
        fixed[:, -1] = anchor_value
        normalized_margin = (
            (margin_at_centers[chosen] - minimum_margin)
            / max(1e-6, 1.0 - minimum_margin)
        ).clamp(0.0, 1.0)
        weight = (
            local_weight[chosen]
            * normalized_margin
            * reliability_at_centers[chosen].clamp_min(0.0)
        )
        keep_anchor = weight > 0
        return nodes[keep_anchor], fixed[keep_anchor], weight[keep_anchor], eligible_count

    local_nodes_padded = torch.cat([
        local_nodes,
        torch.full((local_nodes.shape[0], 1), -2, dtype=torch.long, device=target_features.device),
    ], dim=1)
    local_fixed = torch.zeros_like(local_nodes_padded, dtype=target_features.dtype)
    fg_nodes, fg_fixed, fg_weight, fg_candidate_count = make_anchors(
        (flat_sf - flat_sbn).clamp(-1.0, 1.0),
        1.0,
        flat_view,
        fg_anchor_margin,
        min_fg_view_reliability,
    )
    bg_nodes, bg_fixed, bg_weight, bg_candidate_count = make_anchors(
        (flat_sbn - flat_sf).clamp(-1.0, 1.0),
        0.0,
        torch.ones_like(flat_view),
        bg_anchor_margin,
        0.0,
    )
    edge_nodes = torch.cat([local_nodes_padded, fg_nodes, bg_nodes], dim=0)
    fixed_values = torch.cat([local_fixed, fg_fixed, bg_fixed], dim=0)
    raw_weights = torch.cat([local_weight, fg_weight, bg_weight], dim=0)

    target_mask = edge_nodes >= 0
    degree = torch.zeros(num_nodes, device=target_features.device, dtype=target_features.dtype)
    degree.scatter_add_(0, edge_nodes[target_mask], torch.ones_like(edge_nodes[target_mask], dtype=target_features.dtype))
    safe_nodes = edge_nodes.clamp_min(0)
    edge_degrees = degree[safe_nodes].masked_fill(~target_mask, 0.0)
    edge_degree_mean = edge_degrees.sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)
    weights = raw_weights / edge_degree_mean.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    counts = {
        "num_local_hyperedges": int(local_nodes.shape[0]),
        "num_fg_anchor_hyperedges": int(fg_nodes.shape[0]),
        "num_bg_anchor_hyperedges": int(bg_nodes.shape[0]),
        "num_fg_anchor_candidates": fg_candidate_count,
        "num_bg_anchor_candidates": bg_candidate_count,
    }
    return edge_nodes, fixed_values, weights, degree, counts


def _build_second_order_stencils(
    target_features: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    similarity_threshold: float,
    rgb_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Build fixed three-node stencils for signed second-order TV.

    A stencil is retained only when both adjacent semantic links are coherent
    and neither link crosses a strong RGB discontinuity.  The middle node is
    always the second column of the returned ``[i, j, k]`` triple.
    """
    if not 0.0 <= rgb_quantile <= 1.0:
        raise ValueError("second-order RGB quantile must be in [0, 1]")
    channels, height, width = target_features.shape
    tokens = F.normalize(
        target_features.permute(1, 2, 0).reshape(-1, channels), p=2, dim=1
    )
    rgb = F.interpolate(
        target_rgb.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).reshape(-1, target_rgb.shape[0])
    rows = torch.arange(height, device=target_features.device)
    cols = torch.arange(width, device=target_features.device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    triples: list[torch.Tensor] = []
    similarities: list[torch.Tensor] = []
    rgb_deltas: list[torch.Tensor] = []
    for row_step, col_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        middle_rows = grid_rows
        middle_cols = grid_cols
        valid = (
            (middle_rows - row_step >= 0) & (middle_rows + row_step < height)
            & (middle_cols - col_step >= 0) & (middle_cols + col_step < width)
        )
        middle = (middle_rows * width + middle_cols)[valid]
        first = ((middle_rows - row_step) * width + (middle_cols - col_step))[valid]
        last = ((middle_rows + row_step) * width + (middle_cols + col_step))[valid]
        triples.append(torch.stack([first, middle, last], dim=1))
        similarities.append(torch.stack([
            (tokens[first] * tokens[middle]).sum(dim=1),
            (tokens[middle] * tokens[last]).sum(dim=1),
        ], dim=1))
        rgb_deltas.append(torch.stack([
            (rgb[first] - rgb[middle]).square().sum(dim=1).sqrt(),
            (rgb[middle] - rgb[last]).square().sum(dim=1).sqrt(),
        ], dim=1))
    indices = torch.cat(triples, dim=0)
    pair_similarity = torch.cat(similarities, dim=0)
    pair_rgb_delta = torch.cat(rgb_deltas, dim=0)
    rgb_limit = torch.quantile(pair_rgb_delta.reshape(-1), rgb_quantile)
    keep = (
        (pair_similarity >= similarity_threshold).all(dim=1)
        & (pair_rgb_delta <= rgb_limit).all(dim=1)
    )
    indices = indices[keep]
    if indices.numel() == 0:
        raise RuntimeError("No second-order stencils passed semantic/RGB gating")
    weights = pair_similarity[keep].mean(dim=1).clamp_min(0.0)
    degree = torch.zeros(height * width, device=target_features.device, dtype=target_features.dtype)
    degree.scatter_add_(0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=target_features.dtype))
    stencil_degree = degree[indices].mean(dim=1).clamp_min(1.0)
    weights = weights / stencil_degree
    weights = weights / weights.mean().clamp_min(1e-6)
    return indices, weights, {
        "num_second_order_stencils": int(indices.shape[0]),
        "second_order_mean_weight": float(weights.mean().item()),
        "second_order_rgb_limit": float(rgb_limit.item()),
        "second_order_mean_feature_similarity": float(pair_similarity[keep].mean().item()),
    }


@torch.no_grad()
def hypergraph_tv_refine(
    score: torch.Tensor,
    target_features: torch.Tensor,
    sf: torch.Tensor,
    sbn: torch.Tensor,
    view_reliability: torch.Tensor,
    *,
    lam: float,
    iterations: int,
    local_similarity_threshold: float,
    anchor_ratio: float,
    fg_anchor_margin: float,
    bg_anchor_margin: float,
    min_fg_view_reliability: float,
    primal_step: float,
    dual_step: float,
    tolerance: float,
    second_order: bool = False,
    second_order_lambda: float = 0.01,
    second_order_rgb_quantile: float = 0.75,
    target_rgb: torch.Tensor | None = None,
    evidence_interval: bool = False,
    evidence_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    evidence_interval_max_width: float = 0.15,
    evidence_interval_scale: float = 0.3,
    evidence_interval_epsilon: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Minimize fidelity plus weighted hyperedge range TV with PDHG.

    Every range term is represented as max over ordered node pairs.  Its dual
    variable lies on a scaled simplex, which gives a parameter-free projection
    for each hyperedge and avoids unstructured max/min gradient updates.
    """
    if score.ndim != 2 or target_features.ndim != 3:
        raise ValueError("Expected score [H,W] and target_features [C,H,W]")
    if (
        lam < 0
        or iterations < 1
        or not 0 < anchor_ratio <= 1
        or not -1.0 <= fg_anchor_margin <= 1.0
        or not -1.0 <= bg_anchor_margin <= 1.0
        or not 0.0 <= min_fg_view_reliability <= 1.0
        or evidence_interval_max_width < 0.0
        or evidence_interval_scale < 0.0
        or not 0.0 < evidence_interval_epsilon <= 1.0
        or second_order_lambda < 0.0
        or not 0.0 <= second_order_rgb_quantile <= 1.0
    ):
        raise ValueError("Invalid hypergraph TV configuration")

    height, width = score.shape
    num_nodes = height * width
    s0 = score - score.min()
    s0 = s0 / s0.max().clamp_min(1e-6)
    if lam == 0:
        return s0, {
            "num_hyperedges": 0,
            "num_local_hyperedges": 0,
            "num_fg_anchor_hyperedges": 0,
            "num_bg_anchor_hyperedges": 0,
            "iterations": 0,
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "fallback": False,
            "lambda_zero_identity": True,
        }

    try:
        edge_nodes, fixed_values, weights, degree, edge_counts = _build_hyperedges_fast(
            target_features,
            sf,
            sbn,
            view_reliability,
            similarity_threshold=local_similarity_threshold,
            anchor_ratio=anchor_ratio,
            fg_anchor_margin=fg_anchor_margin,
            bg_anchor_margin=bg_anchor_margin,
            min_fg_view_reliability=min_fg_view_reliability,
        )
    except RuntimeError as error:
        return s0, {
            "num_hyperedges": 0,
            "num_local_hyperedges": 0,
            "num_fg_anchor_hyperedges": 0,
            "num_bg_anchor_hyperedges": 0,
            "iterations": 0,
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "fallback": True,
            "fallback_reason": str(error),
        }
    valid_nodes = edge_nodes >= -1
    num_edges, max_nodes = edge_nodes.shape
    source_slot = torch.arange(max_nodes, device=score.device).view(1, -1, 1)
    destination_slot = torch.arange(max_nodes, device=score.device).view(1, 1, -1)
    pair_valid = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
    pair_valid &= source_slot != destination_slot
    source_index = edge_nodes.unsqueeze(2).expand(-1, -1, max_nodes)
    destination_index = edge_nodes.unsqueeze(1).expand(-1, max_nodes, -1)

    fidelity = 1.0 + 4.0 * (s0.reshape(-1) - 0.5).abs()
    interval_width = None
    interval_disagreement = None
    if evidence_interval:
        if evidence_maps is None or any(item.shape != score.shape for item in evidence_maps):
            raise ValueError("evidence_maps must contain four [H,W] maps when evidence_interval is enabled")
        if not all(torch.isfinite(item).all() for item in evidence_maps):
            raise ValueError("evidence_maps contain non-finite values")
        interval_width, interval_disagreement, _ = _evidence_interval(
            evidence_maps,
            max_width=evidence_interval_max_width,
            scale=evidence_interval_scale,
        )
        lower = (s0 - interval_width).clamp(0.0, 1.0).reshape(-1)
        upper = (s0 + interval_width).clamp(0.0, 1.0).reshape(-1)
    else:
        lower = upper = None
    second_order_indices = None
    second_order_weights = None
    second_order_counts: dict[str, float | int] = {
        "num_second_order_stencils": 0,
        "second_order_mean_weight": 0.0,
        "second_order_rgb_limit": 0.0,
        "second_order_mean_feature_similarity": 0.0,
    }
    if second_order and second_order_lambda > 0.0:
        if target_rgb is None or target_rgb.ndim != 3:
            raise ValueError("target_rgb [C,H,W] is required when second_order is enabled")
        try:
            second_order_indices, second_order_weights, second_order_counts = _build_second_order_stencils(
                target_features,
                target_rgb,
                similarity_threshold=local_similarity_threshold,
                rgb_quantile=second_order_rgb_quantile,
            )
        except RuntimeError as error:
            second_order_counts["second_order_fallback_reason"] = str(error)
    z = s0.reshape(-1).clone()
    z_bar = z.clone()
    dual = torch.zeros((num_edges, max_nodes, max_nodes), device=score.device, dtype=score.dtype)
    pair_mass = (lam * weights).clamp_min(1e-8)
    second_dual = (
        torch.zeros(second_order_indices.shape[0], device=score.device, dtype=score.dtype)
        if second_order_indices is not None else None
    )
    second_mass = (
        second_order_lambda * second_order_weights
        if second_order_weights is not None else None
    )
    primal_residual = float("inf")
    dual_residual = float("inf")

    for iteration in range(iterations):
        node_values = fixed_values.clone()
        target_slots = edge_nodes >= 0
        node_values[target_slots] = z_bar[edge_nodes[target_slots]]
        pair_values = node_values.unsqueeze(2) - node_values.unsqueeze(1)
        old_dual = dual
        dual = _project_scaled_simplex(
            dual + dual_step * pair_values,
            pair_mass,
            pair_valid,
        )

        gradient = torch.zeros_like(z)
        valid_source = (source_index >= 0) & pair_valid
        valid_destination = (destination_index >= 0) & pair_valid
        gradient.scatter_add_(
            0,
            source_index[valid_source],
            dual[valid_source],
        )
        gradient.scatter_add_(
            0,
            destination_index[valid_destination],
            -dual[valid_destination],
        )
        if second_order_indices is not None and second_dual is not None and second_mass is not None:
            stencil_values = (
                z_bar[second_order_indices[:, 0]]
                - 2.0 * z_bar[second_order_indices[:, 1]]
                + z_bar[second_order_indices[:, 2]]
            )
            old_second_dual = second_dual
            second_dual = (second_dual + dual_step * stencil_values).clamp(
                min=-second_mass, max=second_mass,
            )
            gradient.scatter_add_(0, second_order_indices[:, 0], second_dual)
            gradient.scatter_add_(0, second_order_indices[:, 1], -2.0 * second_dual)
            gradient.scatter_add_(0, second_order_indices[:, 2], second_dual)
        else:
            old_second_dual = None
        old_z = z
        proposal = z - primal_step * gradient
        if evidence_interval:
            z = _interval_fidelity_prox(
                proposal, s0.reshape(-1), fidelity, lower, upper,
                primal_step=primal_step, epsilon=evidence_interval_epsilon,
            )
        else:
            z = ((proposal + primal_step * fidelity * s0.reshape(-1)) /
                 (1.0 + primal_step * fidelity)).clamp(0.0, 1.0)
        z_bar = (2.0 * z - old_z).clamp(0.0, 1.0)
        primal_residual = float((z - old_z).abs().max().item())
        dual_residual = float((dual - old_dual).abs().max().item())
        if old_second_dual is not None:
            dual_residual = max(dual_residual, float((second_dual - old_second_dual).abs().max().item()))
        if max(primal_residual, dual_residual) <= tolerance:
            break


    diagnostics = {
        "num_hyperedges": int(num_edges),
        **edge_counts,
        "mean_hyperedge_weight": float(weights.mean().item()),
        "mean_node_degree": float(degree.mean().item()),
        "max_node_degree": float(degree.max().item()),
        "iterations": int(iteration + 1),
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        "fallback": False,
        "lambda_zero_identity": False,
        "mean_absolute_score_change": float((z - s0.reshape(-1)).abs().mean().item()),
        "evidence_interval": bool(evidence_interval),
        "evidence_interval_max_width": float(evidence_interval_max_width),
        "evidence_interval_scale": float(evidence_interval_scale),
        "evidence_interval_epsilon": float(evidence_interval_epsilon),
        "evidence_interval_mean_width": (
            float(interval_width.mean().item()) if interval_width is not None else 0.0
        ),
        "evidence_interval_max_observed_width": (
            float(interval_width.max().item()) if interval_width is not None else 0.0
        ),
        "evidence_interval_mean_disagreement": (
            float(interval_disagreement.mean().item())
            if interval_disagreement is not None else 0.0
        ),
        "second_order": bool(second_order),
        "second_order_lambda": float(second_order_lambda),
        "second_order_rgb_quantile": float(second_order_rgb_quantile),
        "second_order_active": second_order_indices is not None,
        **second_order_counts,
    }
    return z.view(height, width), diagnostics
