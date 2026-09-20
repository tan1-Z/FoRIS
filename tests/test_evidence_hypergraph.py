"""Regression tests for output-preserving evidence-hypergraph helpers."""

import torch

from utils.evidence_hypergraph import (
    ClusterHypergraph,
    CorrespondenceHypergraph,
    prototype_soft_hyperedge_potential,
)


def test_cluster_hypergraph_matches_previous_cluster_reductions():
    labels = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
    values = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])
    candidate = torch.tensor([True, False, True, False, True, False])

    hypergraph = ClusterHypergraph.from_labels(labels)
    old_means = torch.stack([values[labels == k].mean() for k in range(3)])
    old_counts = torch.zeros(3)
    matched_ids, matched_counts = labels[candidate].unique(return_counts=True)
    old_counts[matched_ids] = matched_counts.float()

    assert torch.equal(hypergraph.pool_mean(values), old_means)
    assert torch.equal(
        hypergraph.candidate_counts(candidate, dtype=values.dtype), old_counts,
    )
    assert torch.equal(hypergraph.broadcast(old_means), old_means[labels])


def test_correspondence_hypergraph_matches_mask_lookup_vote():
    first_indices = torch.tensor([[0, 3], [2, 1]], dtype=torch.long)
    second_indices = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    first_membership = torch.tensor([[True, False], [True, False]])
    second_membership = torch.tensor([[False, True], [True, True]])

    old_votes = (
        first_membership.reshape(-1)[first_indices].to(torch.int32)
        + second_membership.reshape(-1)[second_indices].to(torch.int32)
    )
    hypergraph = CorrespondenceHypergraph(
        (first_indices, second_indices), target_shape=(2, 2),
    )

    new_votes = hypergraph.vote_reference_membership(
        (first_membership, second_membership), dtype=torch.float32,
    ).to(torch.int32)
    assert torch.equal(new_votes, old_votes)


def test_soft_prototype_hyperedge_matches_previous_logsumexp():
    torch.manual_seed(0)
    target = torch.randn(1, 4, 2, 3)
    prototypes = torch.randn(5, 4)
    temperature = 0.07

    old_similarity = torch.einsum("bchw,kc->bkhw", target, prototypes)
    old_score = (
        temperature * torch.logsumexp(old_similarity / temperature, dim=1)
    ).squeeze(0)
    new_score, new_similarity = prototype_soft_hyperedge_potential(
        target, prototypes, temperature=temperature,
    )

    assert torch.equal(new_similarity, old_similarity)
    assert torch.equal(new_score, old_score)
