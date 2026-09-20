"""EasyOCR text extraction for document processing."""

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
    """EasyOCR engine for document text extraction."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine = None

    def _load_engine(self):
        """Lazy-load EasyOCR to avoid unnecessary startup overhead."""
        if self._engine is None:
            try:
                import easyocr

                self._engine = easyocr.Reader(
                    self.config.languages,
                    gpu=self.config.use_gpu,
                )

                logger.info("EasyOCR engine loaded successfully")

            except ImportError:
                logger.error(
                    "EasyOCR not installed. "
                    "Install with: pip install easyocr"
                )
                raise

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run EasyOCR on the image."""
        self._load_engine()

        if image is None or image.size == 0:
            logger.warning("Empty image supplied to EasyOCR")

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="easyocr",
            )

        results = self._engine.readtext(image)

        boxes = []

        for bbox, text, confidence in results:
            # EasyOCR returns:
            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

            bbox_int = [
                [int(point[0]), int(point[1])]
                for point in bbox
            ]

            boxes.append(
                OCRBox(
                    text=str(text),
                    confidence=float(confidence),
                    bbox=bbox_int,
                    engine="easyocr",
                )
            )

        full_text = " ".join(
            box.text for box in boxes
        )

        avg_confidence = (
            float(np.mean([box.confidence for box in boxes]))
            if boxes
            else 0.0
        )

        logger.info(
            f"EasyOCR: {len(boxes)} text regions, "
            f"avg confidence: {avg_confidence:.3f}"
        )

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg_confidence,
            engine_used="easyocr",
        )


class OCREngineManager:
    """Manages EasyOCR as the single OCR engine."""

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
    """Draw OCR bounding boxes and text on the image for visualization."""

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
            max(0, int(pts[0][1]) - 5),
        )

        label = f"{box.text} ({box.confidence:.2f})"

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
