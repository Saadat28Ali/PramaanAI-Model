"""Multi-engine OCR text extraction with confidence-based fallback."""

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
        """Run OCR on an image and return structured results."""
        pass


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR wrapper — primary OCR engine (DB + CRNN pipeline)."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine = None

    def _load_engine(self):
        """Lazy-load PaddleOCR to reduce startup time."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
                import logging as _logging
                _logging.getLogger("ppocr").setLevel(_logging.WARNING)
                self._engine = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                )
                logger.info("PaddleOCR engine loaded successfully")
            except ImportError:
                logger.error(
                    "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
                )
                raise

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run PaddleOCR on the image."""
        self._load_engine()

        results = self._engine.predict(image)

        boxes = []
        if results:
            for page in results:
                texts = page.get("rec_texts", [])
                scores = page.get("rec_scores", [])
                polys = page.get("rec_polys", page.get("dt_polys", []))

                for text, score, poly in zip(texts, scores, polys):
                    confidence = float(score)
                    poly_arr = np.asarray(poly)
                    # Convert polygon to 4-point bbox
                    if poly_arr.ndim == 2 and poly_arr.shape[0] >= 4:
                        bbox = [[int(p[0]), int(p[1])] for p in poly_arr[:4]]
                    else:
                        x_min, y_min = int(poly_arr.min(axis=0)[0]), int(poly_arr.min(axis=0)[1])
                        x_max, y_max = int(poly_arr.max(axis=0)[0]), int(poly_arr.max(axis=0)[1])
                        bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

                    boxes.append(OCRBox(
                        text=str(text),
                        confidence=confidence,
                        bbox=bbox,
                        engine="paddleocr",
                    ))

        full_text = " ".join(b.text for b in boxes)
        avg_conf = np.mean([b.confidence for b in boxes]) if boxes else 0.0

        logger.info(
            f"PaddleOCR: {len(boxes)} text regions, avg confidence: {avg_conf:.3f}"
        )

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=float(avg_conf),
            engine_used="paddleocr",
        )


class EasyOCREngine(BaseOCREngine):
    """EasyOCR wrapper — fallback engine (CRAFT + different recognition net)."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine = None

    def _load_engine(self):
        """Lazy-load EasyOCR."""
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
                    "EasyOCR not installed. Install with: pip install easyocr"
                )
                raise

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run EasyOCR on the image."""
        self._load_engine()

        results = self._engine.readtext(image)

        boxes = []
        for (bbox, text, confidence) in results:
            # EasyOCR returns bbox as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
            boxes.append(OCRBox(
                text=text,
                confidence=float(confidence),
                bbox=bbox_int,
                engine="easyocr",
            ))

        full_text = " ".join(b.text for b in boxes)
        avg_conf = np.mean([b.confidence for b in boxes]) if boxes else 0.0

        logger.info(
            f"EasyOCR: {len(boxes)} text regions, avg confidence: {avg_conf:.3f}"
        )

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=float(avg_conf),
            engine_used="easyocr",
        )


class OCREngineManager:
    """Manages multiple OCR engines with confidence-based fallback."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engines = {}

    def _get_engine(self, name: str) -> BaseOCREngine:
        """Get or create an OCR engine by name."""
        if name not in self._engines:
            if name == "paddleocr":
                self._engines[name] = PaddleOCREngine(self.config)
            elif name == "easyocr":
                self._engines[name] = EasyOCREngine(self.config)
            else:
                raise ValueError(f"Unknown OCR engine: {name}")
        return self._engines[name]

    def extract(self, image: np.ndarray) -> OCRResult:
        """Extract text using primary engine, falling back if confidence is low."""
        # Try primary engine
        primary = self._get_engine(self.config.engine)
        try:
            primary_result = primary.extract(image)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Primary OCR engine ({self.config.engine}) failed: {e}")
            primary_result = OCRResult(
                boxes=[], full_text="", avg_confidence=0.0,
                engine_used=self.config.engine,
            )

        if primary_result.avg_confidence >= self.config.confidence_threshold:
            return primary_result

        logger.info(
            f"Primary OCR confidence ({primary_result.avg_confidence:.3f}) "
            f"below threshold ({self.config.confidence_threshold}). "
            f"Trying fallback engine: {self.config.fallback_engine}"
        )

        # Try fallback engine
        fallback = self._get_engine(self.config.fallback_engine)
        try:
            fallback_result = fallback.extract(image)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Fallback OCR engine ({self.config.fallback_engine}) failed: {e}")
            return primary_result

        if fallback_result.avg_confidence > primary_result.avg_confidence:
            logger.info(
                f"Fallback engine ({self.config.fallback_engine}) produced better result: "
                f"{fallback_result.avg_confidence:.3f} vs {primary_result.avg_confidence:.3f}"
            )
            return fallback_result

        return primary_result

    def extract_with_both(self, image: np.ndarray) -> Tuple[OCRResult, OCRResult]:
        """Run both engines and return both results (for benchmarking)."""
        primary = self._get_engine(self.config.engine)
        fallback = self._get_engine(self.config.fallback_engine)

        primary_result = primary.extract(image)
        fallback_result = fallback.extract(image)

        return primary_result, fallback_result


def draw_ocr_boxes(
    image: np.ndarray,
    boxes: List[OCRBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw OCR bounding boxes and text on the image for visualization."""
    vis = image.copy()
    for box in boxes:
        pts = np.array(box.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        text_pos = (pts[0][0], pts[0][1] - 5)
        label = f"{box.text} ({box.confidence:.2f})"
        cv2.putText(
            vis, label, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )

    return vis
