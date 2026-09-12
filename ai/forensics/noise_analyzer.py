"""Noise inconsistency detection via SRM residual patterns."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger


# Standard SRM kernels that suppress image content and amplify noise traces.
# (Bayar & Stamm, IEEE TIFS 2018 used these as constrained first-layer filters.)
SRM_KERNELS: List[np.ndarray] = [
    # 1st-order residual
    np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32),
    # 2nd-order residual
    np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  1, -4,  1,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32),
    # 3rd-order residual
    np.array([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32),
]


@dataclass
class NoiseAnalysisResult:
    """Result of noise inconsistency analysis."""
    noise_map: np.ndarray                        # Per-pixel noise residual (grayscale uint8)
    block_variance_map: np.ndarray               # Per-block variance heatmap (float32)
    inconsistency_score: float                   # 0.0 (uniform) to 1.0 (highly inconsistent)
    suspicious_blocks: List[Dict[str, int | float]]  # blocks with outlier variance
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "inconsistency_score": round(self.inconsistency_score, 4),
            "suspicious_blocks": self.suspicious_blocks,
            "message": self.message,
        }


class NoiseAnalyzer:
    """Detects noise inconsistencies that indicate image splicing."""

    def __init__(
        self,
        block_size: int = 32,
        outlier_sigma: float = 2.5,
    ) -> None:
        self.block_size = block_size
        self.outlier_sigma = outlier_sigma

    def analyze(self, image: np.ndarray) -> NoiseAnalysisResult:
        """Run noise inconsistency analysis on a BGR image."""
        # Extract noise residual via SRM filters
        noise_map = self.extract_noise_residual(image)

        # Per-block variance map
        block_var_map = self._compute_block_variances(noise_map)

        # Score inconsistency
        score = self._compute_inconsistency_score(block_var_map)

        # Find suspicious blocks
        suspicious = self._find_suspicious_blocks(block_var_map)

        message = (
            f"Noise inconsistency DETECTED (score: {score:.3f}). "
            f"{len(suspicious)} suspicious block(s)."
            if score > 0.5
            else f"Noise is consistent (score: {score:.3f})."
        )

        if score > 0.5:
            logger.warning(message)
        else:
            logger.info(message)

        return NoiseAnalysisResult(
            noise_map=noise_map,
            block_variance_map=block_var_map,
            inconsistency_score=score,
            suspicious_blocks=suspicious,
            message=message,
        )

    @staticmethod
    def extract_noise_residual(image: np.ndarray) -> np.ndarray:
        """Extract noise residual by averaging absolute SRM filter responses."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray_f = gray.astype(np.float32)

        responses = []
        for kernel in SRM_KERNELS:
            filtered = cv2.filter2D(gray_f, -1, kernel)
            responses.append(np.abs(filtered))

        combined = np.mean(responses, axis=0)

        # Normalise to uint8, clip at 3σ
        mu = combined.mean()
        sigma = combined.std()
        clip_high = mu + 3 * sigma if sigma > 0 else 255
        normalised = np.clip(combined / max(clip_high, 1e-6) * 255, 0, 255)
        return normalised.astype(np.uint8)

    def _compute_block_variances(self, noise_map: np.ndarray) -> np.ndarray:
        """Compute per-block noise variance over non-overlapping blocks."""
        bs = self.block_size
        h, w = noise_map.shape[:2]
        rows = h // bs
        cols = w // bs
        if rows == 0 or cols == 0:
            return np.zeros((1, 1), dtype=np.float32)

        trimmed = noise_map[:rows * bs, :cols * bs].astype(np.float32)
        blocks = trimmed.reshape(rows, bs, cols, bs)
        return blocks.var(axis=(1, 3))

    def _compute_inconsistency_score(self, block_var_map: np.ndarray) -> float:
        """CV of per-block noise variances, normalised to [0, 1]."""
        mu = block_var_map.mean()
        if mu < 1e-6:
            return 0.0
        cv = block_var_map.std() / mu
        return float(np.clip(cv / 2.0, 0.0, 1.0))

    def _find_suspicious_blocks(
        self, block_var_map: np.ndarray
    ) -> List[Dict[str, int | float]]:
        """Flag blocks whose noise variance is > μ + outlier_sigma × σ."""
        mu = block_var_map.mean()
        sigma = block_var_map.std()
        threshold = mu + self.outlier_sigma * sigma

        suspicious = []
        rows, cols = block_var_map.shape
        for r in range(rows):
            for c in range(cols):
                v = float(block_var_map[r, c])
                if v > threshold:
                    suspicious.append({
                        "bx": int(c * self.block_size),
                        "by": int(r * self.block_size),
                        "block_w": self.block_size,
                        "block_h": self.block_size,
                        "variance": round(v, 2),
                    })

        suspicious.sort(key=lambda b: b["variance"], reverse=True)
        return suspicious

    def visualize(self, block_var_map: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Upsample block variance map to a JET heatmap at image resolution."""
        h, w = image_shape[:2]
        # Normalise to 0-255
        if block_var_map.max() > 0:
            norm = (block_var_map / block_var_map.max() * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(block_var_map, dtype=np.uint8)
        resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap(resized, cv2.COLORMAP_JET)
