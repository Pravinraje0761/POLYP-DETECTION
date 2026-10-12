"""Architecture checks for the paper's segmentation model."""
import pytest
import torch

from polypgate.models import ADenseUNet, ADenseUNet_Boosted

PAPER_PARAM_COUNT = 6_584_870  # Table 15


def test_alias_is_same_class():
    assert ADenseUNet is ADenseUNet_Boosted


def test_parameter_count_matches_paper():
    n = sum(p.numel() for p in ADenseUNet().parameters())
    assert n == PAPER_PARAM_COUNT


@pytest.mark.parametrize("size", [224, 320])
def test_forward_returns_single_channel_logits_at_input_size(size):
    model = ADenseUNet().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 1, size, size)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients():
    model = ADenseUNet().train()
    out = model(torch.randn(1, 3, 128, 128))
    out.mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
