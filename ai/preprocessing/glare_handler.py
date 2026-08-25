"""Glare detection and inpainting for laminated document images."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from loguru import logger

from ..config import GlareConfig


@dataclass
class GlareResult:
    """Result of glare detection and inpainting."""
    has_glare: bool
    glare_ratio: float
    glare_mask: np.ndarray
    inpainted: np.ndarray
    recoverable: bool

    def to_dict(self) -> dict:
        return {
            "has_glare": self.has_glare,
            "glare_ratio": round(self.glare_ratio, 4),
            "recoverable": self.recoverable,
        }


class GlareHandler:
    """Detects and removes glare from document images via HSV thresholding and inpainting."""

    def __init__(self, config: GlareConfig | None = None) -> None:
        self.config = config or GlareConfig()

    def process(self, image: np.ndarray) -> GlareResult:
        """Detect glare and attempt inpainting."""
        glare_mask = self._detect_glare(image)
        total_pixels = image.shape[0] * image.shape[1]
        glare_pixels = np.sum(glare_mask > 0)
        glare_ratio = glare_pixels / total_pixels

        has_glare = glare_ratio > 0.01

        if not has_glare:
            logger.debug("No significant glare detected")
            return GlareResult(
                has_glare=False,
                glare_ratio=glare_ratio,
                glare_mask=glare_mask,
                inpainted=image.copy(),
                recoverable=True,
            )

        recoverable = glare_ratio < 0.30

        if recoverable:
            inpainted = self._inpaint_glare(image, glare_mask)
            logger.info(f"Glare detected ({glare_ratio:.1%}), inpainting applied")
        else:
            inpainted = image.copy()
            logger.warning(
                f"Severe glare ({glare_ratio:.1%}), inpainting may be insufficient"
            )

        return GlareResult(
            has_glare=True,
            glare_ratio=glare_ratio,
            glare_mask=glare_mask,
            inpainted=inpainted,
            recoverable=recoverable,
        )

    def _detect_glare(self, image: np.ndarray) -> np.ndarray:
        """Create a binary mask of glare regions via HSV V-channel thresholding."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        _, glare_mask = cv2.threshold(
            v_channel, self.config.hsv_v_threshold, 255, cv2.THRESH_BINARY
        )

        # Filter small bright spots via connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            glare_mask, connectivity=8
        )

        filtered_mask = np.zeros_like(glare_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.min_glare_area:
                filtered_mask[labels == i] = 255

        # Dilate to cover soft edges of glare
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.dilate_kernel_size, self.config.dilate_kernel_size),
        )
        filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=1)

        return filtered_mask

    def _inpaint_glare(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint glare regions using Telea's Fast Marching method."""
        inpainted = cv2.inpaint(
            image, mask,
            inpaintRadius=self.config.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )
        return inpainted

    def visualize_glare(
        self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Create a visualization with glare regions highlighted in red."""
        vis = image.copy()
        overlay = image.copy()
        overlay[mask > 0] = (0, 0, 255)  # Red overlay on glare
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        return vis
