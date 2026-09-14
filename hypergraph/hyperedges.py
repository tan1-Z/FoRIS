from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HyperEdgeSet:
    edges: list[torch.Tensor]
    weights: list[torch.Tensor]
    relation: list[str]

    def append(self, members: torch.Tensor, weight: torch.Tensor, relation: str) -> None:
        if members.numel() >= 2 and torch.isfinite(weight) and float(weight) > 0:
            self.edges.append(members.long())
            self.weights.append(weight)
            self.relation.append(relation)


def _chunked_topk(query: torch.Tensor, key: torch.Tensor, k: int, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine TopK without retaining a full query×key similarity matrix."""
    all_scores, all_indices = [], []
    for start in range(0, query.shape[0], chunk):
        score = query[start:start + chunk] @ key.T
        values, indices = torch.topk(score, k=min(k, key.shape[0]), dim=1)
        all_scores.append(values)
        all_indices.append(indices)
    return torch.cat(all_scores), torch.cat(all_indices)


def build_relation_hyperedges(
    ref_feat: torch.Tensor,
    query_feat: torch.Tensor,
    ref_occupancy: torch.Tensor,
    query_hw: tuple[int, int],
    *,
    semantic_k: int,
    spatial_radius: int,
    cross_topk: int,
    semantic_threshold: float,
    background_modes: int,
    use_topology: bool,
) -> tuple[HyperEdgeSet, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build semantic, spatial, cross-image and background hyperedges.

    Nodes are ordered as all reference patches followed by query patches.
    The returned query seed tensors are [Nq] foreground/background confidence.
    """
    ref_feat = F.normalize(ref_feat, p=2, dim=1)
    query_feat = F.normalize(query_feat, p=2, dim=1)
    nr, nq = ref_feat.shape[0], query_feat.shape[0]
    edge_set = HyperEdgeSet([], [], [])

    # Query semantic groups: anchor plus feature-nearest query patches.
    sem_scores, sem_indices = _chunked_topk(query_feat, query_feat, semantic_k + 1)
    for anchor in range(nq):
        keep = sem_scores[anchor] >= semantic_threshold
        members = sem_indices[anchor, keep] + nr
        if members.numel() >= 3:
            edge_set.append(members, sem_scores[anchor, keep].mean().clamp_min(1e-6), "semantic")

    # Spatial groups use a local grid and require feature-compatible members.
    height, width = query_hw
    for row in range(height):
        for col in range(width):
            r0, r1 = max(0, row - spatial_radius), min(height, row + spatial_radius + 1)
            c0, c1 = max(0, col - spatial_radius), min(width, col + spatial_radius + 1)
            rr, cc = torch.meshgrid(
                torch.arange(r0, r1, device=query_feat.device),
                torch.arange(c0, c1, device=query_feat.device), indexing="ij",
            )
            local = (rr * width + cc).reshape(-1)
            anchor = row * width + col
            sim = query_feat[local] @ query_feat[anchor]
            members = local[sim >= semantic_threshold] + nr
            if members.numel() >= 3:
                edge_set.append(members, sim[sim >= semantic_threshold].mean().clamp_min(1e-6), "spatial")

    fg_ref = torch.nonzero(ref_occupancy >= 0.25, as_tuple=False).flatten()
    bg_ref = torch.nonzero(ref_occupancy <= 0.05, as_tuple=False).flatten()
    fg_seed = torch.zeros(nq, device=query_feat.device, dtype=query_feat.dtype)
    bg_seed = torch.zeros_like(fg_seed)

    # Each reference FG node forms a cross-image group with its strongest query matches.
    if fg_ref.numel() > 0:
        fg_scores, fg_indices = _chunked_topk(ref_feat[fg_ref], query_feat, cross_topk)
        for idx, ref_idx in enumerate(fg_ref):
            scores, q_indices = fg_scores[idx], fg_indices[idx]
            members = torch.cat((ref_idx.view(1), q_indices + nr))
            margin = scores[0] - scores[1] if scores.numel() > 1 else scores[0]
            confidence = ((scores.mean() + 1.0) * 0.5 * margin.clamp_min(0)).clamp(0, 1)
            edge_set.append(members, confidence.clamp_min(1e-6), "cross_fg")
            fg_seed.scatter_reduce_(0, q_indices, confidence.expand_as(q_indices), reduce="amax", include_self=True)

    # Multiple reference background modes are represented by groups rather than one mean prototype.
    if bg_ref.numel() > 0:
        mode_count = min(background_modes, bg_ref.numel())
        stride = max(1, bg_ref.numel() // mode_count)
        mode_ref = bg_ref[::stride][:mode_count]
        bg_scores, bg_indices = _chunked_topk(ref_feat[mode_ref], query_feat, cross_topk)
        for idx, ref_idx in enumerate(mode_ref):
            scores, q_indices = bg_scores[idx], bg_indices[idx]
            members = torch.cat((ref_idx.view(1), q_indices + nr))
            confidence = ((scores.mean() + 1.0) * 0.5).clamp(0, 1)
            edge_set.append(members, confidence.clamp_min(1e-6), "background")
            bg_seed.scatter_reduce_(0, q_indices, confidence.expand_as(q_indices), reduce="amax", include_self=True)

    # First stable topology relation: directional local groups, disabled by default.
    if use_topology:
        for row in range(height):
            for col in range(width):
                anchor = row * width + col
                horizontal = torch.tensor([anchor - 1, anchor, anchor + 1], device=query_feat.device)
                horizontal = horizontal[(horizontal >= row * width) & (horizontal < (row + 1) * width)]
                if horizontal.numel() == 3:
                    sim = (query_feat[horizontal] @ query_feat[anchor]).mean()
                    edge_set.append(horizontal + nr, sim.clamp_min(1e-6), "topology")

    counts = {name: edge_set.relation.count(name) for name in set(edge_set.relation)}
    return edge_set, fg_seed, bg_seed, counts
