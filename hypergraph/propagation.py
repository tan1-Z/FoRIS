from __future__ import annotations

import torch


def propagate_labels(
    initial: torch.Tensor,
    reference_nodes: torch.Tensor,
    edges: list[torch.Tensor],
    weights: torch.Tensor,
    *,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    """Two-channel sparse node↔hyperedge label propagation.

    Args:
        initial: [N, 2] foreground/background seed states.
        reference_nodes: [N] true at reference patch nodes.
        edges: variable-size node-index tensors.
        weights: [E] non-negative relation-normalized edge weights.
    """
    state = initial.clone()
    for _ in range(steps):
        node_sum = torch.zeros_like(state)
        node_degree = torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
        for members, weight in zip(edges, weights):
            if members.numel() < 2 or float(weight) <= 0:
                continue
            edge_state = state[members].mean(dim=0) * weight
            node_sum.index_add_(0, members, edge_state.expand(members.numel(), -1))
            node_degree.index_add_(0, members, weight.expand(members.numel(), 1))
        propagated = node_sum / node_degree.clamp_min(1e-8)
        state = alpha * propagated + (1.0 - alpha) * initial
        state[reference_nodes] = initial[reference_nodes]
        state = state.clamp_min(0)
    return state
