"""Behavioural checks for the segmentation loss and metrics."""
import torch

from polypgate.losses import CombinedLoss, FocalTverskyLoss
from polypgate.metrics import compute_segmentation_metrics


def _square_mask(size=64):
    mask = torch.zeros(1, 1, size, size)
    mask[..., 16:48, 16:48] = 1.0
    return mask


def test_combined_loss_defaults_match_paper():
    loss = CombinedLoss()
    ft = loss.focal_tversky
    assert (ft.alpha, ft.beta, ft.gamma) == (0.3, 0.7, 0.75)


def test_loss_prefers_correct_prediction():
    target = _square_mask()
    good_logits = (target * 2 - 1) * 10  # confident and correct
    bad_logits = -good_logits            # confident and wrong
    loss = CombinedLoss()
    assert loss(good_logits, target) < loss(bad_logits, target)


def test_focal_tversky_penalises_false_negatives_more_than_false_positives():
    # beta > alpha, so missing polyp pixels should cost more than over-segmenting.
    target = _square_mask()
    loss = FocalTverskyLoss()
    missed = torch.full_like(target, -10.0)                       # predicts nothing
    over = torch.full_like(target, 10.0)                          # predicts everything
    assert loss(missed, target) > loss(over, target)


def test_metrics_perfect_prediction():
    target = _square_mask()
    m = compute_segmentation_metrics(target.clone(), target)
    for key in ("dice", "iou", "precision", "recall", "specificity", "accuracy"):
        assert abs(m[key] - 1.0) < 1e-4, key


def test_metrics_known_overlap():
    target = _square_mask()                 # 32x32 = 1024 positive pixels
    pred = torch.zeros_like(target)
    pred[..., 16:48, 16:32] = 1.0           # exactly half of them
    m = compute_segmentation_metrics(pred, target)
    assert abs(m["dice"] - 2 / 3) < 1e-4    # 2*512 / (512 + 1024)
    assert abs(m["iou"] - 0.5) < 1e-4
    assert abs(m["recall"] - 0.5) < 1e-4
    assert abs(m["precision"] - 1.0) < 1e-4
