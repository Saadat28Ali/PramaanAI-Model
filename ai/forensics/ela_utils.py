"""Shared ELA computation — single source of truth for recompression-based ELA."""

from __future__ import annotations

import cv2
import numpy as np


def compute_ela_map(
    image: np.ndarray,
    quality: int = 95,
    scale_factor: int = 15,
) -> np.ndarray:
    """Compute an Error Level Analysis map.

    Re-encodes at *quality* via in-memory JPEG round-trip, takes the
    pixel-wise diff, averages across channels, and amplifies by
    *scale_factor*. Returns a grayscale uint8 map.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(image, recompressed)
    gray_diff = np.mean(diff.astype(np.float32), axis=2)
    amplified = gray_diff * scale_factor
    return np.clip(amplified, 0, 255).astype(np.uint8)
