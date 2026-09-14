import torch

from hypergraph.propagation import propagate_labels


def test_foreground_reference_supports_query_nodes() -> None:
    initial = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    reference = torch.tensor([True, False, False])
    state = propagate_labels(
        initial, reference, [torch.tensor([0, 1, 2])], torch.tensor([1.0]),
        steps=3, alpha=0.9,
    )
    assert state[1, 0] > 0
    assert state[2, 0] > 0
    assert torch.equal(state[0], initial[0])


def test_background_reference_lowers_foreground_difference() -> None:
    initial = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    state = propagate_labels(
        initial, torch.tensor([True, False]), [torch.tensor([0, 1])], torch.tensor([1.0]),
        steps=2, alpha=0.9,
    )
    assert state[1, 1] > state[1, 0]
