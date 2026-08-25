"""
Copy-Move Forgery Detection — Keypoint-based duplicate region finder.

Copy-move forgery pastes one region of an image onto another region of
the *same* image.  Detection exploits the fact that duplicated regions
share local feature descriptors.  This module uses:

    1.  ORB keypoints (fast, rotation-invariant)
    2.  BFMatcher (Hamming) with ratio test
    3.  RANSAC geometric verification — filters matches that don't
        form a coherent affine transform

Only match pairs that are spatially *distant* survive: nearby matches
are self-matches from regular texture / edges.

References:
    - Amerini et al., "A SIFT-Based Forensic Method for Copy-Move
      Attack Detection and Transformation Recovery",
      IEEE TIFS 2011.
    - Christlein et al., "An Evaluation of Popular Copy-Move Forgery
      Detection Approaches", IEEE TIFS 2012.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from loguru import logger


@dataclass
class CopyMoveResult:
    """Result of copy-move detection analysis."""
    is_copymove: bool
    match_count: int
    confidence: float
    matched_regions: List[Tuple[int, int, int, int]]
    visualisation: np.ndarray
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_copymove": self.is_copymove,
            "match_count": self.match_count,
            "confidence": round(self.confidence, 4),
            "matched_regions": [
                {"x": r[0], "y": r[1], "w": r[2], "h": r[3]}
                for r in self.matched_regions
            ],
            "message": self.message,
        }


class CopyMoveDetector:
    """Detects copy-move forgery via ORB features and RANSAC filtering."""

    def __init__(
        self,
        n_features: int = 5000,
        ratio_thresh: float = 0.75,
        min_distance: int = 50,
        ransac_reproj: float = 5.0,
        min_inliers: int = 15,
    ) -> None:
        self.n_features = n_features
        self.ratio_thresh = ratio_thresh
        self.min_distance = min_distance
        self.ransac_reproj = ransac_reproj
        self.min_inliers = min_inliers

    def detect(self, image: np.ndarray) -> CopyMoveResult:
        """Run copy-move detection on a BGR image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=self.n_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return self._empty_result(image, "Too few keypoints for copy-move analysis.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(descriptors, descriptors, k=3)

        # k=3: best match is self (d=0), use 2nd & 3rd for ratio test
        good_pairs: List[Tuple[int, int]] = []
        for m_list in raw_matches:
            if len(m_list) < 3:
                continue
            m1, m2 = m_list[1], m_list[2]
            if m1.distance < self.ratio_thresh * m2.distance:
                good_pairs.append((m1.queryIdx, m1.trainIdx))

        if len(good_pairs) < self.min_inliers:
            return self._empty_result(image, f"Only {len(good_pairs)} feature matches — below threshold.")

        src_pts: List[np.ndarray] = []
        dst_pts: List[np.ndarray] = []
        for qi, ti in good_pairs:
            p1 = np.array(keypoints[qi].pt)
            p2 = np.array(keypoints[ti].pt)
            dist = np.linalg.norm(p1 - p2)
            if dist > self.min_distance:
                src_pts.append(p1)
                dst_pts.append(p2)

        if len(src_pts) < self.min_inliers:
            return self._empty_result(image, f"Only {len(src_pts)} distant matches — below threshold.")

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)

        _, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj
        )

        if mask is None:
            return self._empty_result(image, "RANSAC failed — no consistent transform.")

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < self.min_inliers:
            return self._empty_result(image, f"Only {n_inliers} RANSAC inliers — below threshold.")

        inlier_src = src[inlier_mask]
        inlier_dst = dst[inlier_mask]
        regions = self._cluster_to_rects(inlier_src, inlier_dst)

        # Scale from 0 at min_inliers to 1.0 at 5× min_inliers
        confidence = float(np.clip(n_inliers / (self.min_inliers * 5), 0.0, 1.0))

        vis = self._visualize(image, inlier_src, inlier_dst, regions)

        message = (
            f"Copy-move DETECTED — {n_inliers} inlier matches, "
            f"confidence {confidence:.2f}, {len(regions)} region cluster(s)."
        )
        logger.warning(message)

        return CopyMoveResult(
            is_copymove=True,
            match_count=n_inliers,
            confidence=confidence,
            matched_regions=regions,
            visualisation=vis,
            message=message,
        )


    def _empty_result(self, image: np.ndarray, msg: str) -> CopyMoveResult:
        logger.info(f"Copy-move: {msg}")
        return CopyMoveResult(
            is_copymove=False,
            match_count=0,
            confidence=0.0,
            matched_regions=[],
            visualisation=image.copy(),
            message=msg,
        )

    @staticmethod
    def _cluster_to_rects(
        src_pts: np.ndarray, dst_pts: np.ndarray, pad: int = 15
    ) -> List[Tuple[int, int, int, int]]:
        """Compute bounding rectangles for source and destination point clouds."""
        rects: List[Tuple[int, int, int, int]] = []
        for pts in (src_pts, dst_pts):
            x_min, y_min = pts.min(axis=0).astype(int) - pad
            x_max, y_max = pts.max(axis=0).astype(int) + pad
            rects.append((
                max(int(x_min), 0),
                max(int(y_min), 0),
                int(x_max - x_min),
                int(y_max - y_min),
            ))
        return rects

    @staticmethod
    def _visualize(
        image: np.ndarray,
        inlier_src: np.ndarray,
        inlier_dst: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Draw match lines and bounding rectangles on a copy of the image."""
        vis = image.copy()

        # Draw match lines
        for (sx, sy), (dx, dy) in zip(inlier_src, inlier_dst):
            pt1 = (int(sx), int(sy))
            pt2 = (int(dx), int(dy))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 0, 255), -1)

        # Draw bounding rectangles
        colours = [(255, 0, 0), (0, 0, 255)]
        for i, (x, y, w, h) in enumerate(regions):
            cv2.rectangle(vis, (x, y), (x + w, y + h), colours[i % 2], 2)

        return vis
