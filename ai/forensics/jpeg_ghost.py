"""
JPEG Ghost Analysis — Detects double-compression artifacts from splicing.

When an image is saved as JPEG at quality Q₁ and then a spliced region
(from a different JPEG at quality Q₂) is pasted in and re-saved at Q₃,
the spliced region has a *different* quantisation history than its
surroundings.  JPEG ghost analysis sweeps across quality levels and
finds the quality at which each region shows minimum re-compression error
— the "ghost" quality reveals the original compression level.

This module implements:
    1. Multi-quality re-compression sweep (Q = 50..98)
    2. Per-block minimum-error quality estimation
    3. Ghost inconsistency scoring (CV of per-block ghost qualities)

References:
    - Farid, "Exposing Digital Forgeries from JPEG Ghosts",
      IEEE TIFS 2009.
    - Pan et al., "Exposing Image Forgery with Blind Noise Estimation",
      ACM Multimedia 2011.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger


@dataclass
class JPEGGhostResult:
    """Result of JPEG ghost analysis."""
    ghost_quality_map: np.ndarray
    ghost_heatmap: np.ndarray
    inconsistency_score: float
    dominant_quality: int
    suspicious_blocks: List[Dict[str, int | float]]
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "inconsistency_score": round(self.inconsistency_score, 4),
            "dominant_quality": self.dominant_quality,
            "suspicious_blocks": self.suspicious_blocks,
            "message": self.message,
        }


class JPEGGhostDetector:
    """Detects double-JPEG compression via ghost quality analysis."""

    def __init__(
        self,
        block_size: int = 16,
        q_low: int = 50,
        q_high: int = 98,
        q_step: int = 2,
    ) -> None:
        self.block_size = block_size
        self.q_low = q_low
        self.q_high = q_high
        self.q_step = q_step

    def analyze(self, image: np.ndarray) -> JPEGGhostResult:
        """Run JPEG ghost analysis on a BGR image."""
        qualities = list(range(self.q_low, self.q_high + 1, self.q_step))
        bs = self.block_size
        h, w = image.shape[:2]
        rows = h // bs
        cols = w // bs

        if rows == 0 or cols == 0:
            return JPEGGhostResult(
                ghost_quality_map=np.zeros((1, 1), dtype=np.float32),
                ghost_heatmap=np.zeros_like(image),
                inconsistency_score=0.0,
                dominant_quality=95,
                suspicious_blocks=[],
                message="Image too small for JPEG ghost analysis.",
            )

        # Trim to block multiples
        h_t, w_t = rows * bs, cols * bs
        trimmed = image[:h_t, :w_t]

        # error_volume shape: (len(qualities), rows, cols)
        error_volume = np.zeros((len(qualities), rows, cols), dtype=np.float64)

        for qi, q in enumerate(qualities):
            _, encoded = cv2.imencode(".jpg", trimmed, [cv2.IMWRITE_JPEG_QUALITY, q])
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            # Per-pixel absolute difference (grayscale mean)
            diff = np.mean(
                cv2.absdiff(trimmed, recompressed).astype(np.float64), axis=2
            )

            # Aggregate to blocks
            blocks = diff.reshape(rows, bs, cols, bs)
            error_volume[qi] = blocks.mean(axis=(1, 3))

        best_q_idx = error_volume.argmin(axis=0)  # (rows, cols)
        ghost_quality_map = np.array(qualities)[best_q_idx].astype(np.float32)

        q_flat = ghost_quality_map.ravel().astype(int)
        counts = np.bincount(q_flat - self.q_low)
        dominant_quality = int(counts.argmax() + self.q_low)

        mu = ghost_quality_map.mean()
        score = 0.0
        if mu > 0:
            cv_val = ghost_quality_map.std() / mu
            score = float(np.clip(cv_val * 5.0, 0.0, 1.0))

        suspicious: List[Dict[str, int | float]] = []
        for r in range(rows):
            for c in range(cols):
                q_val = int(ghost_quality_map[r, c])
                if abs(q_val - dominant_quality) > 6:
                    suspicious.append({
                        "bx": int(c * bs),
                        "by": int(r * bs),
                        "block_w": bs,
                        "block_h": bs,
                        "ghost_quality": q_val,
                        "dominant_quality": dominant_quality,
                    })

        suspicious.sort(key=lambda b: abs(b["ghost_quality"] - dominant_quality), reverse=True)

        ghost_heatmap = self._visualize(ghost_quality_map, (h, w))

        message = (
            f"JPEG ghost inconsistency DETECTED (score: {score:.3f}). "
            f"Dominant Q={dominant_quality}, {len(suspicious)} outlier block(s)."
            if score > 0.3
            else f"JPEG compression is consistent (Q≈{dominant_quality}, score: {score:.3f})."
        )

        if score > 0.3:
            logger.warning(message)
        else:
            logger.info(message)

        return JPEGGhostResult(
            ghost_quality_map=ghost_quality_map,
            ghost_heatmap=ghost_heatmap,
            inconsistency_score=score,
            dominant_quality=dominant_quality,
            suspicious_blocks=suspicious,
            message=message,
        )

    def _visualize(self, quality_map: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Map per-block ghost qualities to a JET heatmap at image resolution."""
        h, w = target_shape[:2]
        q_range = self.q_high - self.q_low
        if q_range == 0:
            q_range = 1
        norm = ((quality_map - self.q_low) / q_range * 255).astype(np.uint8)
        resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap(resized, cv2.COLORMAP_JET)
