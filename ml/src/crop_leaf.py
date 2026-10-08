"""Leaf cropping before classification — fixes domain shift (whole-plant field photos).

Pure functions only; OpenCV is the single dependency. No model loading.
Contract: crop -> caller resizes with Lanczos to 256x256, feeds 0..255 (see class_names.json).
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

GREEN_LO = np.array([25, 40, 25], dtype=np.uint8)  # HSV lower bound for leaf green
GREEN_HI = np.array([95, 255, 255], dtype=np.uint8)
MIN_AREA_RATIO = 0.02  # reject masks covering <2% of frame (noise)


class CropResult(NamedTuple):
    image: np.ndarray  # cropped BGR image
    ok: bool  # False -> use original image (crop failed or not helpful)
    box: tuple[int, int, int, int] | None  # x, y, w, h
    mask_coverage: float


def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of green leaf pixels with morphological cleanup."""
    if bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Keep the largest connected green component (the leaf), None if too small."""
    if mask.size == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < MIN_AREA_RATIO * mask.size:
        return None
    return (labels == idx).astype(np.uint8) * 255


def crop_to_mask(bgr: np.ndarray, mask: np.ndarray, pad: float = 0.12) -> CropResult:
    """Crop image to masked object bbox with padding. ok=False if coverage too low."""
    if bgr.size == 0:
        return CropResult(bgr, False, None, 0.0)
    ys, xs = np.where(mask > 0)
    coverage = float(len(xs)) / (bgr.shape[0] * bgr.shape[1])
    if len(xs) == 0 or coverage < MIN_AREA_RATIO:
        return CropResult(bgr, False, None, coverage)
    h, w = bgr.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    px, py = int(pad * (x1 - x0)), int(pad * (y1 - y0))
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    return CropResult(bgr[y0:y1, x0:x1], True, (x0, y0, x1 - x0, y1 - y0), coverage)


def crop_leaf(bgr: np.ndarray) -> CropResult:
    """Full crop pipeline: green mask -> largest leaf -> bbox crop. Never raises."""
    mask = largest_component(leaf_mask(bgr))
    if mask is None:
        return CropResult(bgr, False, None, 0.0)
    return crop_to_mask(bgr, mask)


def lanczos_resize(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Enforce the inference contract: Lanczos resize to size x size (see class_names.json)."""
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def prepare_for_model(bgr: np.ndarray, use_crop: bool = True) -> np.ndarray:
    """Field photo -> model input (uint8/float 0..255, HxW=size). Returns HxWx3."""
    cropped = crop_leaf(bgr).image if use_crop else bgr
    return lanczos_resize(cropped, 256).astype(np.float32)