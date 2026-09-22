"""Lightweight EasyOCR text extraction for document processing."""

import cv2
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from ..config import OCRConfig


@dataclass
class OCRBox:
    """A single detected text region."""

    text: str
    confidence: float
    bbox: List[List[int]]
    engine: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "engine": self.engine,
        }


@dataclass
class OCRResult:
    """Complete OCR result for an image."""

    boxes: List[OCRBox]
    full_text: str
    avg_confidence: float
    engine_used: str
    raw_output: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "boxes": [b.to_dict() for b in self.boxes],
            "full_text": self.full_text,
            "avg_confidence": round(self.avg_confidence, 4),
            "engine_used": self.engine_used,
        }


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> OCRResult:
        """Run OCR on the image and return structured results."""
        pass


class EasyOCREngine(BaseOCREngine):
    """
    Lightweight EasyOCR engine for document text extraction.

    The OCR model is loaded lazily and the input image is resized
    before OCR to reduce memory usage on resource-constrained
    deployments such as Render.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine = None

        # Maximum dimension sent to EasyOCR.
        # This prevents very large document images from causing
        # excessive memory usage.
        self.max_dimension = 1600

        # Ignore very low-confidence OCR results.
        self.min_confidence = 0.20

        # Ignore extremely short OCR detections.
        self.min_text_length = 2

    def _load_engine(self):
        """Lazy-load a lightweight CPU EasyOCR engine."""

        if self._engine is not None:
            return

        try:
            import easyocr

            logger.info("Loading lightweight EasyOCR engine...")

            self._engine = easyocr.Reader(
                self.config.languages,
                gpu=False,
                quantize=True,
                verbose=False,
            )

            logger.info("EasyOCR engine loaded successfully")

        except ImportError:
            logger.error(
                "EasyOCR not installed. "
                "Install with: pip install easyocr"
            )
            raise

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image before OCR to reduce memory consumption.

        The original image is preserved outside this method.
        """

        height, width = image.shape[:2]

        largest_dimension = max(height, width)

        if largest_dimension <= self.max_dimension:
            return image

        scale = self.max_dimension / float(largest_dimension)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        logger.info(
            f"Resizing OCR image: "
            f"{width}x{height} -> "
            f"{new_width}x{new_height}"
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run lightweight EasyOCR on the supplied image."""

        self._load_engine()

        # --------------------------------------------------
        # Validate image
        # --------------------------------------------------

        if image is None or image.size == 0:
            logger.warning("Empty image supplied to EasyOCR")

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="easyocr",
            )

        # --------------------------------------------------
        # Prepare smaller OCR image
        # --------------------------------------------------

        image_for_ocr = self._prepare_image(image)

        logger.info(
            f"EasyOCR input size: "
            f"{image_for_ocr.shape[1]}x"
            f"{image_for_ocr.shape[0]}"
        )

        # --------------------------------------------------
        # Run OCR
        # --------------------------------------------------

        try:
            results = self._engine.readtext(
                image_for_ocr,
                detail=1,
                paragraph=False,
                batch_size=1,
                mag_ratio=1.0,
            )

        except Exception as e:
            logger.error(f"EasyOCR inference failed: {e}")
            raise

        # --------------------------------------------------
        # Convert OCR results
        # --------------------------------------------------

        boxes: List[OCRBox] = []

        for bbox, text, confidence in results:

            text = str(text).strip()
            confidence = float(confidence)

            # Ignore empty / extremely short detections.
            if len(text) < self.min_text_length:
                continue

            # Ignore very low-confidence detections.
            if confidence < self.min_confidence:
                continue

            bbox_int = [
                [
                    int(point[0]),
                    int(point[1]),
                ]
                for point in bbox
            ]

            boxes.append(
                OCRBox(
                    text=text,
                    confidence=confidence,
                    bbox=bbox_int,
                    engine="easyocr",
                )
            )

        # --------------------------------------------------
        # Build combined text
        # --------------------------------------------------

        full_text = " ".join(
            box.text
            for box in boxes
        )

        # --------------------------------------------------
        # Calculate average confidence
        # --------------------------------------------------

        if boxes:
            avg_confidence = float(
                np.mean(
                    [
                        box.confidence
                        for box in boxes
                    ]
                )
            )
        else:
            avg_confidence = 0.0

        logger.info(
            f"EasyOCR: {len(boxes)} relevant text regions, "
            f"avg confidence: {avg_confidence:.3f}"
        )

        # --------------------------------------------------
        # Return structured OCR result
        # --------------------------------------------------

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg_confidence,
            engine_used="easyocr",
        )


class OCREngineManager:
    """
    Manages EasyOCR as the single OCR engine.

    The engine itself is created lazily so that the OCR model
    is not loaded unless OCR is actually requested.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine = None

    def _get_engine(self) -> EasyOCREngine:
        """Get or create the EasyOCR engine."""

        if self._engine is None:
            self._engine = EasyOCREngine(self.config)

        return self._engine

    def extract(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR only."""

        engine = self._get_engine()

        try:
            return engine.extract(image)

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"EasyOCR failed: {e}")

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="easyocr",
            )

    def extract_with_both(
        self,
        image: np.ndarray,
    ) -> Tuple[OCRResult, OCRResult]:
        """
        Kept for backward compatibility.

        EasyOCR is now the only OCR engine, so both returned
        results are the same OCR result.
        """

        result = self.extract(image)

        return result, result


def draw_ocr_boxes(
    image: np.ndarray,
    boxes: List[OCRBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw OCR bounding boxes and text on the image."""

    vis = image.copy()

    for box in boxes:

        pts = np.array(
            box.bbox,
            dtype=np.int32,
        )

        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

        text_pos = (
            int(pts[0][0]),
            max(
                0,
                int(pts[0][1]) - 5,
            ),
        )

        label = (
            f"{box.text} "
            f"({box.confidence:.2f})"
        )

        cv2.putText(
            vis,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    return vis
