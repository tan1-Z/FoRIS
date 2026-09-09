import torch
import torch.nn as nn

from models.foris import FoRIS
from utils.data import downsample_mask_soft


class DummyEncoder(nn.Module):
    """Small DINO-shaped encoder: one intermediate call can return many blocks."""
    n_blocks = 24

    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_intermediate_layers(self, x, n=1, reshape=True):
        self.calls += 1
        ids = list(range(self.n_blocks - int(n), self.n_blocks)) if isinstance(n, int) else list(n)
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h // 8), torch.linspace(0, 1, w // 8), indexing="ij")
        base = torch.stack([yy, xx, yy * xx, yy + xx, yy - xx, xx * xx, yy * yy, torch.ones_like(yy)]).to(x)
        return [(base * (i + 1)).unsqueeze(0).expand(b, -1, -1, -1).contiguous() for i in ids]


def make_model(**kwargs):
    return FoRIS(DummyEncoder(), image_size=32, svd_components=4, device="cpu", **kwargs)


def test_multilevel_features_one_encoder_call():
    model = make_model(adaptive_multilayer=True, multilayer_ids=[11, 15, 19, 23])
    model.encoder.calls = 0
    out = model._extract_multilevel_features(torch.randn(1, 2, 3, 32, 32))
    assert list(out) == [11, 15, 19, 23]
    assert all(v.shape == (1, 2, 8, 4, 4) for v in out.values())
    assert model.encoder.calls == 1


def test_soft_mask_preserves_thin_line():
    m = torch.zeros(1, 1, 32, 32)
    m[:, :, 15, :] = 1
    soft = downsample_mask_soft(m, 4, 4)
    assert soft.sum() > 0
    assert ((soft > 0) & (soft < 1)).any()


def test_router_is_normal_and_valid():
    model = make_model(adaptive_multilayer=True)
    metrics = {11: {"separability": .1, "compactness": .2, "boundary": .1, "shot_consistency": 0.},
               15: {"separability": .2, "compactness": .1, "boundary": .4, "shot_consistency": 0.},
               19: {"separability": .4, "compactness": .3, "boundary": .2, "shot_consistency": 0.}}
    weights, fl, fc = model._route_layers(metrics, .7)
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert fl in metrics and fc in metrics


def test_roi_budget_and_empty_local_blend():
    model = make_model(local_refine=True, local_refine_topk=2, local_refine_area_budget=.25)
    p = torch.zeros(4, 4); p[0, 0] = 1; p[3, 3] = .9
    rois = model._propose_refine_rois(p, (32, 32))
    assert len(rois) <= 2
    assert sum((r["bbox"][2]-r["bbox"][0])*(r["bbox"][3]-r["bbox"][1]) for r in rois) <= 256
    score = torch.rand(32, 32)
    assert torch.equal(model._blend_local_refinement(score, torch.rand(3, 32, 32), [], 23, torch.rand(8)), score)

