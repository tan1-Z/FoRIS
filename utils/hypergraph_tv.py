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


def _build_hyperedges_fast(
    target_features: torch.Tensor,
    sf: torch.Tensor,
    sbn: torch.Tensor,
    view_reliability: torch.Tensor,
    *,
    similarity_threshold: float,
    anchor_ratio: float,
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

    def make_anchors(confidence: torch.Tensor, anchor_value: float, reliability: torch.Tensor):
        confidence_at_centers = confidence[local_centers]
        count = min(anchors, confidence_at_centers.numel())
        chosen = confidence_at_centers.topk(count).indices
        nodes = torch.cat([
            local_nodes[chosen],
            torch.full((count, 1), -1, dtype=torch.long, device=target_features.device),
        ], dim=1)
        fixed = torch.zeros((count, 6), device=target_features.device, dtype=target_features.dtype)
        fixed[:, -1] = anchor_value
        weight = (
            local_weight[chosen]
            * confidence_at_centers[chosen].clamp_min(0.0)
            * reliability[local_centers[chosen]].clamp_min(0.0)
        )
        keep_anchor = weight > 0
        return nodes[keep_anchor], fixed[keep_anchor], weight[keep_anchor]

    local_nodes_padded = torch.cat([
        local_nodes,
        torch.full((local_nodes.shape[0], 1), -2, dtype=torch.long, device=target_features.device),
    ], dim=1)
    local_fixed = torch.zeros_like(local_nodes_padded, dtype=target_features.dtype)
    fg_nodes, fg_fixed, fg_weight = make_anchors(
        (flat_sf * (1.0 - flat_sbn)).clamp(0.0, 1.0), 1.0, flat_view
    )
    bg_nodes, bg_fixed, bg_weight = make_anchors(
        (flat_sbn * (1.0 - flat_sf)).clamp(0.0, 1.0), 0.0, torch.ones_like(flat_view)
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
    }
    return edge_nodes, fixed_values, weights, degree, counts


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
    primal_step: float,
    dual_step: float,
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Minimize fidelity plus weighted hyperedge range TV with PDHG.

    Every range term is represented as max over ordered node pairs.  Its dual
    variable lies on a scaled simplex, which gives a parameter-free projection
    for each hyperedge and avoids unstructured max/min gradient updates.
    """
    if score.ndim != 2 or target_features.ndim != 3:
        raise ValueError("Expected score [H,W] and target_features [C,H,W]")
    if lam < 0 or iterations < 1 or not 0 < anchor_ratio <= 1:
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
    z = s0.reshape(-1).clone()
    z_bar = z.clone()
    dual = torch.zeros((num_edges, max_nodes, max_nodes), device=score.device, dtype=score.dtype)
    pair_mass = (lam * weights).clamp_min(1e-8)
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
        old_z = z
        proposal = z - primal_step * gradient
        z = ((proposal + primal_step * fidelity * s0.reshape(-1)) /
             (1.0 + primal_step * fidelity)).clamp(0.0, 1.0)
        z_bar = (2.0 * z - old_z).clamp(0.0, 1.0)
        primal_residual = float((z - old_z).abs().max().item())
        dual_residual = float((dual - old_dual).abs().max().item())
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
    }
    return z.view(height, width), diagnostics
