"""Quality gate — evaluate blur, exposure, resolution, and glare before pipeline entry."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from loguru import logger

from ..config import QualityGateConfig


@dataclass
class QualityReport:
    """Result of the quality gate evaluation."""
    passed: bool
    blur_score: float
    tenengrad_score: float
    brightness: float
    contrast: float
    resolution: Tuple[int, int]
    glare_ratio: float
    issues: List[str]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "blur_score": round(self.blur_score, 2),
            "tenengrad_score": round(self.tenengrad_score, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "resolution": list(self.resolution),
            "glare_ratio": round(self.glare_ratio, 4),
            "issues": self.issues,
        }


class QualityGate:
    """Multi-dimensional image quality check with actionable failure messages."""

    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(self, image: np.ndarray) -> QualityReport:
        """Run all quality checks and return a QualityReport."""
        issues = []

        # Resolution
        h, w = image.shape[:2]
        resolution = (w, h)
        if min(h, w) < self.config.min_resolution:
            issues.append(
                f"Resolution too low: {w}x{h}. Minimum dimension: {self.config.min_resolution}px."
            )

        # Blur (Laplacian variance)
        blur_score = self._compute_blur_score(image)
        if blur_score < self.config.blur_threshold:
            issues.append(
                f"Image is blurry (Laplacian score: {blur_score:.1f}, "
                f"threshold: {self.config.blur_threshold}). "
                "Hold the camera steady."
            )

        # Secondary focus (Tenengrad)
        tenengrad_score = self._compute_tenengrad(image)
        tenengrad_thresh = getattr(self.config, "tenengrad_threshold", 300.0)
        if tenengrad_score < tenengrad_thresh:
            issues.append(
                f"Poor focus (Tenengrad: {tenengrad_score:.1f}, "
                f"threshold: {tenengrad_thresh:.0f}). "
                "Ensure the camera is focused on the document."
            )

        # Brightness / exposure
        brightness = self._compute_brightness(image)
        if brightness < self.config.min_brightness:
            issues.append(
                f"Image is too dark (brightness: {brightness:.1f}). "
                "Move to a well-lit area."
            )
        elif brightness > self.config.max_brightness:
            issues.append(
                f"Image is overexposed (brightness: {brightness:.1f}). "
                "Reduce direct light on the document."
            )

        contrast = self._compute_contrast(image)

        # Glare
        glare_ratio = self._compute_glare_ratio(image)
        if glare_ratio > self.config.max_glare_ratio:
            issues.append(
                f"Excessive glare detected ({glare_ratio:.1%} of image). "
                "Tilt the document to reduce reflections."
            )

        passed = len(issues) == 0
        if passed:
            logger.info("Quality gate PASSED")
        else:
            logger.warning(f"Quality gate FAILED: {issues}")

        return QualityReport(
            passed=passed,
            blur_score=blur_score,
            tenengrad_score=tenengrad_score,
            brightness=brightness,
            contrast=contrast,
            resolution=resolution,
            glare_ratio=glare_ratio,
            issues=issues,
        )

    def _compute_blur_score(self, image: np.ndarray) -> float:
        """Laplacian variance — low variance indicates blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _compute_tenengrad(self, image: np.ndarray) -> float:
        """Mean Sobel gradient magnitude (Tenengrad focus metric)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = gx ** 2 + gy ** 2
        return float(np.mean(magnitude))

    def _compute_brightness(self, image: np.ndarray) -> float:
        """Mean brightness from the HSV V channel."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        return float(np.mean(v_channel))

    def _compute_contrast(self, image: np.ndarray) -> float:
        """Contrast as standard deviation of grayscale intensity."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def _compute_glare_ratio(self, image: np.ndarray) -> float:
        """Fraction of pixels exceeding the glare intensity threshold."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        glare_mask = (v_channel >= self.config.glare_intensity_threshold).astype(np.uint8)

        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)

        glare_pixels = np.sum(glare_mask)
        total_pixels = image.shape[0] * image.shape[1]

        return float(glare_pixels / total_pixels)

    def get_glare_mask(self, image: np.ndarray) -> np.ndarray:
        """Binary mask (0/255) of glare regions."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        glare_mask = np.zeros_like(v_channel)
        glare_mask[v_channel >= self.config.glare_intensity_threshold] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
        # Dilate slightly to ensure we cover edges of glare
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

        return glare_mask

    def check_black_frame(self, image: np.ndarray, threshold: float = 15.0) -> bool:
        """Check if the image is essentially black (camera covered/off)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < threshold
