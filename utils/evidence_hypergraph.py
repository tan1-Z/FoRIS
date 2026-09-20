"""Exact hypergraph/factor reparameterizations for FoRIS evidence aggregation.

These helpers deliberately retain the existing FoRIS arithmetic.  They make
the set structure explicit without introducing learned weights, message
passing, or a new decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClusterHypergraph:
    """Target patches as vertices and agglomerative clusters as hyperedges.

    ``labels[i] == k`` is the sparse binary incidence relation between target
    vertex ``i`` and semantic-cluster hyperedge ``k``.  Pooling intentionally
    follows the former per-cluster boolean-index reduction order so this class
    is an output-preserving reparameterization, not a new propagation rule.
    """

    labels: torch.Tensor
    num_hyperedges: int

    @classmethod
    def from_labels(cls, labels: torch.Tensor) -> "ClusterHypergraph":
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("cluster labels must be a non-empty 1-D tensor")
        if labels.dtype != torch.long:
            labels = labels.long()
        num_hyperedges = int(labels.max().item()) + 1
        return cls(labels=labels, num_hyperedges=num_hyperedges)

    def member_counts(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Return the cardinality of every semantic-cluster hyperedge."""
        return torch.bincount(
            self.labels, minlength=self.num_hyperedges,
        ).to(dtype=dtype)

    def pool_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Pool one scalar evidence value per vertex into hyperedge means."""
        flat = values.reshape(-1)
        if flat.numel() != self.labels.numel():
            raise ValueError("values and cluster labels must have the same size")
        pooled = torch.zeros(
            self.num_hyperedges, device=flat.device, dtype=flat.dtype,
        )
        for edge_id in range(self.num_hyperedges):
            members = self.labels == edge_id
            if bool(members.any()):
                pooled[edge_id] = flat[members].mean()
        return pooled

    def candidate_counts(self, candidate_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Compute H^T c for a binary candidate signal c over target vertices."""
        candidate = candidate_mask.reshape(-1).bool()
        if candidate.numel() != self.labels.numel():
            raise ValueError("candidate mask and cluster labels must have the same size")
        counts = torch.zeros(
            self.num_hyperedges, device=self.labels.device, dtype=dtype,
        )
        selected = self.labels[candidate]
        if selected.numel() > 0:
            counts.scatter_add_(
                0,
                selected,
                torch.ones(selected.numel(), device=counts.device, dtype=dtype),
            )
        return counts

    def broadcast(self, hyperedge_values: torch.Tensor) -> torch.Tensor:
        """Return H v: assign each vertex its semantic-cluster value."""
        if hyperedge_values.ndim != 1 or hyperedge_values.numel() != self.num_hyperedges:
            raise ValueError("hyperedge_values must contain one value per cluster")
        return hyperedge_values[self.labels]


@dataclass(frozen=True)
class CorrespondenceHypergraph:
    """One typed correspondence factor per target patch across reference shots.

    For a shot, the factor retains exactly the original target-to-reference
    top-1 index.  Its vote operation is therefore identical to mask lookup and
    majority voting in the former implementation.  In one-shot inference this
    factor contains only a target/reference pair and is not a higher-order
    relation; it is kept for a common multi-shot representation.
    """

    matched_reference_indices: tuple[torch.Tensor, ...]
    target_shape: tuple[int, int]

    def vote_reference_membership(
        self,
        reference_memberships: tuple[torch.Tensor, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(reference_memberships) != len(self.matched_reference_indices):
            raise ValueError("every correspondence shot needs a reference membership map")
        height, width = self.target_shape
        votes = torch.zeros((height, width), dtype=torch.int32, device=self.matched_reference_indices[0].device)
        for matched, membership in zip(self.matched_reference_indices, reference_memberships):
            if matched.shape != (height, width):
                raise ValueError("correspondence index shape differs from target shape")
            votes += membership.reshape(-1)[matched].to(torch.int32)
        return votes.to(dtype=dtype)


def prototype_soft_hyperedge_potential(
    target_features: torch.Tensor,
    foreground_prototypes: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the original LogSumExp prototype factor exactly.

    A target patch and all foreground prototypes form one soft factor.  The
    function intentionally returns the original potential rather than replacing
    it with incidence averaging or hypergraph message passing.
    """
    similarity = torch.einsum("bchw,kc->bkhw", target_features, foreground_prototypes)
    safe_temperature = max(1e-4, float(temperature))
    potential = (
        safe_temperature * torch.logsumexp(similarity / safe_temperature, dim=1)
    ).squeeze(0)
    return potential, similarity
