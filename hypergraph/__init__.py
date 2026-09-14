"""Sparse, episode-adaptive hypergraph utilities for HyperFoRIS."""

from .hyperedges import HyperEdgeSet, build_relation_hyperedges
from .propagation import propagate_labels

__all__ = ["HyperEdgeSet", "build_relation_hyperedges", "propagate_labels"]
