"""Perspective correction for document images via 4-point homography."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from loguru import logger

from ..config import RectificationConfig


@dataclass
class RectificationResult:
    """Result of the rectification process."""
    success: bool
    rectified: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    method_used: str
    message: str

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "method_used": self.method_used,
            "message": self.message,
        }


class DocumentRectifier:
    """Detects document boundaries and applies perspective correction."""

    def __init__(self, config: RectificationConfig | None = None) -> None:
        self.config = config or RectificationConfig()

    def rectify(self, image: np.ndarray) -> RectificationResult:
        """Detect document corners and apply perspective transform."""
        for strategy, detect_fn in [
            ("contour", self._detect_corners_contour),
            ("hough", self._detect_corners_hough),
        ]:
            corners = detect_fn(image)
            if corners is None:
                continue

            ok, reason = self._validate_corners(corners, image.shape)
            if not ok:
                logger.debug(f"{strategy} corners rejected: {reason}")
                continue

            rectified = self._apply_perspective_transform(image, corners)
            logger.info(f"Rectification succeeded via {strategy} detection")
            return RectificationResult(
                success=True,
                rectified=rectified,
                corners=corners,
                method_used=strategy,
                message=f"Document rectified using {strategy} method.",
            )

        logger.warning("Rectification failed: no valid document boundary detected")
        return RectificationResult(
            success=False,
            rectified=None,
            corners=None,
            method_used="none",
            message=(
                "Could not detect document boundaries. "
                "Ensure the ID card is fully visible against a contrasting background."
            ),
        )


    def _validate_corners(
        self, corners: np.ndarray, image_shape: tuple
    ) -> tuple[bool, str]:
        """Check convexity, aspect ratio, and reprojection error."""
        # Convexity
        if not cv2.isContourConvex(corners.astype(np.int32)):
            return False, "non-convex quadrilateral"

        # Aspect ratio (ID card ≈ 1.58:1)
        w_top = float(np.linalg.norm(corners[1] - corners[0]))
        w_bot = float(np.linalg.norm(corners[2] - corners[3]))
        h_left = float(np.linalg.norm(corners[3] - corners[0]))
        h_right = float(np.linalg.norm(corners[2] - corners[1]))
        avg_w = (w_top + w_bot) / 2.0
        avg_h = (h_left + h_right) / 2.0
        if avg_h < 1:
            return False, "degenerate height"

        # Normalise so aspect >= 1
        aspect = max(avg_w, avg_h) / min(avg_w, avg_h)
        if aspect < self.config.min_aspect_ratio:
            return False, f"aspect ratio {aspect:.2f} < min {self.config.min_aspect_ratio}"
        if aspect > self.config.max_aspect_ratio:
            return False, f"aspect ratio {aspect:.2f} > max {self.config.max_aspect_ratio}"

        # Reprojection error
        dst = np.array([
            [0, 0],
            [avg_w - 1, 0],
            [avg_w - 1, avg_h - 1],
            [0, avg_h - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        M_inv = cv2.getPerspectiveTransform(dst, corners)

        projected = cv2.perspectiveTransform(
            corners.reshape(1, -1, 2), M
        )
        back = cv2.perspectiveTransform(projected, M_inv).reshape(4, 2)
        errors = np.linalg.norm(back - corners, axis=1)
        max_err = float(errors.max())

        if max_err > self.config.max_reprojection_error:
            return False, f"reprojection error {max_err:.2f}px > threshold"

        return True, "ok"

    def _detect_corners_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect document corners via contour finding and polygon approximation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        edges = cv2.Canny(
            blurred, self.config.canny_low, self.config.canny_high
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.config.min_area_ratio

        for contour in contours[:5]:  # Check top 5 largest
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour, self.config.approx_epsilon_ratio * peri, True
            )

            # We need exactly 4 vertices
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                corners = self._order_points(corners)
                return corners

        return None

    def _detect_corners_hough(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback corner detection via Hough lines and their intersections."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=50, maxLineGap=20
        )

        if lines is None or len(lines) < 4:
            return None

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(angle) < 30 or abs(angle) > 150:
                horizontal_lines.append(line[0])
            elif 60 < abs(angle) < 120:
                vertical_lines.append(line[0])

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None

        h_sorted = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)

        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]
        left_line = v_sorted[0]
        right_line = v_sorted[-1]

        # Compute intersections of the 4 boundary lines
        corners = []
        for h_line in [top_line, bottom_line]:
            for v_line in [left_line, right_line]:
                pt = self._line_intersection(h_line, v_line)
                if pt is not None:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        corners = np.array(corners, dtype=np.float32)
        corners = self._order_points(corners)

        area = cv2.contourArea(corners)
        image_area = image.shape[0] * image.shape[1]
        if area < image_area * self.config.min_area_ratio:
            return None

        return corners

    def _line_intersection(
        self, line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute the intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        rect[1] = pts[np.argmin(d)]   # top-right
        rect[3] = pts[np.argmax(d)]   # bottom-left

        return rect

    def _apply_perspective_transform(
        self, image: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Apply 4-point perspective warp to produce a flat document view."""
        # Compute the width of the new image
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        max_height = max(int(height_left), int(height_right))

        # Clamp to target dimensions while preserving aspect ratio
        target_w = min(max_width, self.config.target_width)
        target_h = min(max_height, self.config.target_height)

        if max_width > 0 and max_height > 0:
            aspect = max_width / max_height
            if target_w / target_h > aspect:
                target_w = int(target_h * aspect)
            else:
                target_h = int(target_w / aspect)

        # Destination points: a perfect rectangle
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ], dtype=np.float32)

        # Compute and apply the perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, dst)
        rectified = cv2.warpPerspective(
            image, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rectified

    def draw_corners(
        self, image: np.ndarray, corners: np.ndarray, color=(0, 255, 0), thickness=3
    ) -> np.ndarray:
        """Draw detected corners and edges on the image for visualization."""
        vis = image.copy()
        pts = corners.astype(int)

        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness)
            cv2.circle(vis, tuple(pts[i]), 8, (0, 0, 255), -1)

        labels = ["TL", "TR", "BR", "BL"]
        for i, label in enumerate(labels):
            cv2.putText(
                vis, label, tuple(pts[i] + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )

        return vis
