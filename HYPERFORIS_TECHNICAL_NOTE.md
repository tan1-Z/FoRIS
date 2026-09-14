# HyperFoRIS technical note

## Implemented first prototype

HyperFoRIS is selected with `--method hyperforis`; original FoRIS remains the
default. It uses the same frozen DINOv3 encoder, reference/query episode API,
dataset pipeline, optional CRF, and mIoU evaluation protocol.

Each episode constructs patch nodes for all reference and query images. The
reference mask is downsampled with area interpolation, preserving soft patch
occupancy in `[0,1]`. Foreground and background states are initialized from
reference occupancy and confidence-weighted cross-image matches. Sparse
semantic, spatial, reference-FG cross-image, and reference-background
hyperedges then propagate two channels: foreground and background.

## Files

- `models/hyperforis.py`: episode API, frozen feature extraction, node labels,
  relation weights, propagation call, and mask reconstruction.
- `hypergraph/hyperedges.py`: semantic, spatial, cross-image foreground,
  background, and optional local directional topology hyperedge builders.
- `hypergraph/propagation.py`: sparse index-based node-to-edge and
  edge-to-node two-channel propagation with clamped reference labels.
- `opts.py`, `models/__init__.py`, `inference.py`: `--method hyperforis`,
  hypergraph parameters, model construction, and independent evaluation JSON.
- `tests/test_hypergraph_propagation.py`: foreground and background label
  propagation sanity tests.

## Current limits

This is the Stage-1 prototype. It implements the four core relation families
but does not yet fuse multiple DINO layers, compute reference geometry
descriptors, route relation weights from geometry, perform coarse-to-fine
inference, or perform target-presence rejection. The corresponding CLI options
for layer selection are logged as reserved research controls and should not be
interpreted as active adaptive layer fusion.

The next experiment should compare original FoRIS with HyperFoRIS semantic-only,
then add spatial, cross-image foreground, and background relations one at a
time. Topology should be evaluated separately on thin-structure datasets.
