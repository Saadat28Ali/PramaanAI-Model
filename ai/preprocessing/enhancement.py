"""Document-specific image enhancement for OCR accuracy."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from ..config import EnhancementConfig


class ImageEnhancer:
    """Document-aware image enhancer optimised for OCR on ID cards."""

    def __init__(self, config: EnhancementConfig | None = None) -> None:
        self.config = config or EnhancementConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )


    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Full enhancement pipeline — CLAHE, denoise, sharpen."""
        enhanced = self.apply_clahe(image)
        enhanced = self.denoise(enhanced)
        enhanced = self.sharpen(enhanced)

        focus = self.compute_tenengrad(enhanced)
        logger.debug(f"Enhancement done | Tenengrad focus: {focus:.1f}")
        return enhanced


    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_ch)
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Non-Local Means denoising (edge-preserving)."""
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.config.denoise_h,
            self.config.denoise_h,
            self.config.denoise_template_window,
            self.config.denoise_search_window,
        )

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Unsharp mask to boost text edges."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


    def binarize_sauvola(self, image: np.ndarray) -> np.ndarray:
        """Sauvola adaptive binarization for text extraction under uneven lighting."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        win = self.config.sauvola_window_size
        k = self.config.sauvola_k
        R = 128.0

        # Local mean and std via box filter (O(1) per pixel)
        gray_f = gray.astype(np.float64)
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(win, win))
        sq_mean = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(win, win))
        std = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None))

        threshold = mean * (1.0 + k * (std / R - 1.0))

        binary = np.zeros_like(gray)
        binary[gray_f >= threshold] = 255

        return binary

    def compute_tenengrad(self, image: np.ndarray) -> float:
        """Tenengrad focus measure — Sobel gradient magnitude variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = gx ** 2 + gy ** 2
        return float(np.mean(magnitude))

    def binarize_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian thresholding (fast alternative to Sauvola)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Gray-world white balance."""
        result = image.astype(np.float32)
        means = result.mean(axis=(0, 1))  # (B, G, R)
        global_mean = means.mean()
        scale = global_mean / (means + 1e-6)
        result *= scale[np.newaxis, np.newaxis, :]
        return np.clip(result, 0, 255).astype(np.uint8)
