"""Checks for the image-enhancement stages."""
import cv2
import numpy as np

from polypgate.preprocessing.clahe import lab_clahe_enhancement
from polypgate.preprocessing.rgb import white_balance_correction


def _colour_cast_image(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(64, 64, 3)).astype(np.uint8)
    img[..., 0] = np.clip(img[..., 0].astype(int) + 50, 0, 255)  # strong red cast, like mucosa
    return img


def test_clahe_preserves_shape_and_dtype():
    img = _colour_cast_image()
    _, out = lab_clahe_enhancement(img)  # returns (before, after)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_clahe_only_changes_luminance():
    # CLAHE is applied to L only, so chrominance (a, b) should be nearly unchanged.
    img = _colour_cast_image()
    _, out = lab_clahe_enhancement(img)  # returns (before, after)
    ab_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 1:].astype(int)
    ab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 1:].astype(int)
    assert np.abs(ab_in - ab_out).mean() < 3


def test_gray_world_white_balance_equalises_channel_means():
    # This is the transform the paper identifies as the cause of the probability collapse.
    img = _colour_cast_image()
    out = white_balance_correction(img).astype(float)
    means = out.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 3
    before = img.reshape(-1, 3).astype(float).mean(axis=0)
    assert before.max() - before.min() > 30
