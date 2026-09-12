"""Error Level Analysis (ELA) for JPEG tamper detection."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from loguru import logger

from ..config import ELAConfig


@dataclass
class ELAResult:
    """Result of Error Level Analysis."""
    anomaly_score: float
    is_tampered: bool
    ela_map: np.ndarray
    heatmap: np.ndarray
    suspicious_regions: List[Dict[str, int | float]]
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "anomaly_score": round(self.anomaly_score, 4),
            "is_tampered": self.is_tampered,
            "suspicious_regions": self.suspicious_regions,
            "message": self.message,
        }


class ELADetector:
    """Performs Error Level Analysis to detect image tampering."""

    def __init__(self, config: ELAConfig | None = None) -> None:
        self.config = config or ELAConfig()

    def analyze(self, image: np.ndarray) -> ELAResult:
        """Perform ELA on the input image."""
        from .ela_utils import compute_ela_map

        ela_map = compute_ela_map(
            image,
            quality=self.config.jpeg_quality,
            scale_factor=self.config.scale_factor,
        )

        anomaly_score = self._compute_anomaly_score(ela_map)
        suspicious_regions = self._find_suspicious_regions(ela_map)
        heatmap = self._generate_heatmap(ela_map)

        is_tampered = anomaly_score > self.config.anomaly_threshold

        message = (
            f"Tamper DETECTED (score: {anomaly_score:.3f}). "
            f"Found {len(suspicious_regions)} suspicious region(s)."
            if is_tampered
            else f"No tampering detected (score: {anomaly_score:.3f})."
        )

        if is_tampered:
            logger.warning(message)
        else:
            logger.info(message)

        return ELAResult(
            anomaly_score=anomaly_score,
            is_tampered=is_tampered,
            ela_map=ela_map,
            heatmap=heatmap,
            suspicious_regions=suspicious_regions,
            message=message,
        )

    def _compute_anomaly_score(self, ela_map: np.ndarray) -> float:
        """
        Composite anomaly score from three signals:
        - Coefficient of variation (CV) of the ELA map
        - Hot-pixel ratio (pixels > μ + 3σ)
        - DCT block variance (8×8 block-level energy non-uniformity)

        Weights are configurable via ``ELAConfig``.
        """
        mean_val = float(np.mean(ela_map))
        std_val = float(np.std(ela_map))

        if mean_val < 1e-6:
            return 0.0

        # CV: natural images ~0.3–0.6, tampered push to 1.5+
        cv_score = min((std_val / mean_val) / 5.0, 1.0)

        # Hot-pixel ratio (μ + 3σ to avoid statistical tail noise)
        hot_thresh = mean_val + 3.0 * std_val
        hot_ratio = float(np.sum(ela_map > hot_thresh)) / ela_map.size
        hot_score = min(hot_ratio * 10.0, 1.0)

        # DCT block energy non-uniformity
        dct_score = self._compute_dct_nonuniformity(ela_map)

        # Weighted combination
        w_cv = self.config.cv_weight
        w_hot = self.config.hot_ratio_weight
        w_dct = self.config.dct_weight
        raw_score = w_cv * cv_score + w_hot * hot_score + w_dct * dct_score

        # Dampen score when overall ELA energy is low (clean single-compressed
        # JPEGs have mean ELA ≈ 0–5 at scale 15)
        intensity_gate = float(min(mean_val / 15.0, 1.0))
        score = raw_score * intensity_gate

        return float(np.clip(score, 0.0, 1.0))

    def _compute_dct_nonuniformity(self, ela_map: np.ndarray) -> float:
        """Measure energy non-uniformity at the JPEG 8×8 block grid.

        Returns normalised score in [0, 1].
        """
        bs = self.config.dct_block_size
        h, w = ela_map.shape[:2]
        # Trim to integer multiples of block size
        h_trim = (h // bs) * bs
        w_trim = (w // bs) * bs
        if h_trim == 0 or w_trim == 0:
            return 0.0

        trimmed = ela_map[:h_trim, :w_trim].astype(np.float64)
        blocks = trimmed.reshape(h_trim // bs, bs, w_trim // bs, bs)
        block_means = blocks.mean(axis=(1, 3))  # (rows, cols)

        mu = block_means.mean()
        if mu < 1e-6:
            return 0.0
        cv = block_means.std() / mu
        return float(min(cv / 3.0, 1.0))

    def _find_suspicious_regions(self, ela_map: np.ndarray) -> List[Dict[str, int | float]]:
        """Identify suspicious regions using thresholding and connected components."""
        # μ + 3.5σ threshold to limit false positives from the normal tail
        mean_val = np.mean(ela_map)
        std_val = np.std(ela_map)
        threshold = mean_val + 3.5 * std_val

        _, binary = cv2.threshold(
            ela_map, int(threshold), 255, cv2.THRESH_BINARY
        )

        # Morphological closing to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = ela_map.shape[0] * ela_map.shape[1] * 0.005

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            region_intensity = float(np.mean(ela_map[y : y + h, x : x + w]))

            regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "intensity": round(region_intensity, 2),
            })

        # Sort by intensity (most suspicious first)
        regions.sort(key=lambda r: r["intensity"], reverse=True)
        return regions

    def _generate_heatmap(self, ela_map: np.ndarray) -> np.ndarray:
        """Convert grayscale ELA map to a JET colourmap heatmap."""
        heatmap = cv2.applyColorMap(ela_map, cv2.COLORMAP_JET)
        return heatmap

    def overlay_heatmap(
        self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Blend the ELA heatmap onto the original image."""
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        return cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)

    def draw_suspicious_regions(
        self, image: np.ndarray, regions: list, color=(0, 0, 255), thickness=2
    ) -> np.ndarray:
        """Draw bounding boxes around suspicious regions."""
        vis = image.copy()
        for region in regions:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            label = f"Anomaly: {region['intensity']:.1f}"
            cv2.putText(
                vis, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return vis
