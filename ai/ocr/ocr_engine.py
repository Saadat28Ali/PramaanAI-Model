"""Lightweight RapidOCR text extraction for document processing."""

import cv2
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from ..config import OCRConfig


# ============================================================
# OCR DATA STRUCTURES
# ============================================================


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


# ============================================================
# BASE OCR ENGINE
# ============================================================


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> OCRResult:
        """Run OCR on the image and return structured results."""
        pass


# ============================================================
# RAPIDOCR ENGINE
# ============================================================


class RapidOCREngine(BaseOCREngine):
    """
    Lightweight RapidOCR engine for document text extraction.

    Uses ONNX Runtime instead of PyTorch/PaddlePaddle.

    The OCR engine is initialized lazily so the models are only
    loaded when OCR is actually requested.
    """

    def __init__(
        self,
        config: OCRConfig | None = None,
    ) -> None:

        self.config = config or OCRConfig()

        self._engine = None

        # Maximum dimension sent to OCR.
        #
        # This is intentionally kept at 1600 to avoid sending
        # unnecessarily large document images into the OCR model.
        self.max_dimension = 1600

        # Ignore extremely low-confidence detections.
        self.min_confidence = 0.20

        # Ignore extremely short detections.
        self.min_text_length = 2

    # ========================================================
    # LOAD RAPIDOCR
    # ========================================================

    def _load_engine(self) -> None:
        """Lazy-load RapidOCR."""

        if self._engine is not None:
            return

        try:

            from rapidocr import RapidOCR

            logger.info(
                "Loading RapidOCR engine..."
            )

            self._engine = RapidOCR()

            logger.info(
                "RapidOCR engine loaded successfully"
            )

        except ImportError:

            logger.error(
                "RapidOCR is not installed. "
                "Install with: "
                "pip install rapidocr onnxruntime"
            )

            raise

        except Exception as e:

            logger.error(
                f"RapidOCR initialization failed: {e}"
            )

            raise

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    def _prepare_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Resize image before OCR.

        Returns:
            prepared_image,
            scale_used
        """

        height, width = image.shape[:2]

        largest_dimension = max(
            height,
            width,
        )

        scale = 1.0

        if largest_dimension > self.max_dimension:

            scale = (
                self.max_dimension
                / float(largest_dimension)
            )

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            logger.info(
                f"Resizing OCR image: "
                f"{width}x{height} -> "
                f"{new_width}x{new_height}"
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        return image, scale

    # ========================================================
    # BBOX CONVERSION
    # ========================================================

    def _convert_bbox(
        self,
        bbox,
        scale: float,
    ) -> List[List[int]]:
        """
        Convert RapidOCR polygon coordinates back to the
        original image coordinate system.

        RapidOCR normally returns:

        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]
        """

        if scale <= 0:
            scale = 1.0

        converted = []

        for point in bbox:

            x = int(
                float(point[0])
                / scale
            )

            y = int(
                float(point[1])
                / scale
            )

            converted.append(
                [x, y]
            )

        return converted

    # ========================================================
    # OCR EXTRACTION
    # ========================================================

    def extract(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """Run RapidOCR on the supplied image."""

        self._load_engine()

        # ====================================================
        # VALIDATE IMAGE
        # ====================================================

        if image is None or image.size == 0:

            logger.warning(
                "Empty image supplied to RapidOCR"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="rapidocr",
            )

        # ====================================================
        # PREPARE IMAGE
        # ====================================================

        image_for_ocr, scale = self._prepare_image(
            image
        )

        logger.info(
            f"RapidOCR input size: "
            f"{image_for_ocr.shape[1]}x"
            f"{image_for_ocr.shape[0]}"
        )

        # ====================================================
        # RUN OCR
        # ====================================================

        try:

            result = self._engine(
                image_for_ocr
            )

        except Exception as e:

            logger.error(
                f"RapidOCR inference failed: {e}"
            )

            raise

        # ====================================================
        # HANDLE EMPTY RESULT
        # ====================================================

        if result is None:

            logger.warning(
                "RapidOCR returned no result"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="rapidocr",
            )

        # ====================================================
        # RAPIDOCR RESULT FORMAT
        # ====================================================
        #
        # Depending on the RapidOCR version, result may
        # expose:
        #
        #   result.boxes
        #   result.txts
        #   result.scores
        #
        # or behave like:
        #
        #   [boxes, texts, scores]
        #
        # We handle both forms.
        # ====================================================

        boxes_raw = None
        texts_raw = None
        scores_raw = None

        # ----------------------------------------------------
        # New/result-object style
        # ----------------------------------------------------

        if hasattr(result, "boxes"):

            boxes_raw = result.boxes

        elif hasattr(result, "polys"):

            boxes_raw = result.polys

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if hasattr(result, "txts"):

            texts_raw = result.txts

        elif hasattr(result, "texts"):

            texts_raw = result.texts

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        if hasattr(result, "scores"):

            scores_raw = result.scores

        # ----------------------------------------------------
        # Tuple/list style
        # ----------------------------------------------------

        if (
            boxes_raw is None
            and isinstance(result, (tuple, list))
            and len(result) >= 3
        ):

            boxes_raw = result[0]
            texts_raw = result[1]
            scores_raw = result[2]

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if (
            boxes_raw is None
            or texts_raw is None
            or scores_raw is None
        ):

            logger.warning(
                "RapidOCR returned an unexpected result format"
            )

            logger.debug(
                f"RapidOCR result type: "
                f"{type(result)}"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="rapidocr",
            )

        # ====================================================
        # CONVERT RESULTS
        # ====================================================

        boxes: List[OCRBox] = []

        for bbox, text, confidence in zip(
            boxes_raw,
            texts_raw,
            scores_raw,
        ):

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            if len(text) < self.min_text_length:
                continue

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            try:

                confidence = float(
                    confidence
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

            if confidence < self.min_confidence:
                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            try:

                bbox_converted = (
                    self._convert_bbox(
                        bbox,
                        scale,
                    )
                )

            except Exception as e:

                logger.warning(
                    f"Could not convert OCR bbox: {e}"
                )

                continue

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            boxes.append(
                OCRBox(
                    text=text,
                    confidence=confidence,
                    bbox=bbox_converted,
                    engine="rapidocr",
                )
            )

        # ====================================================
        # SORT BOXES
        # ====================================================
        #
        # Top -> bottom
        # Left -> right
        #
        # This keeps full_text in a sensible document order.
        # ====================================================

        boxes.sort(
            key=lambda box: (
                box.bbox[0][1],
                box.bbox[0][0],
            )
        )

        # ====================================================
        # BUILD FULL TEXT
        # ====================================================

        full_text = " ".join(
            box.text
            for box in boxes
        )

        # ====================================================
        # AVERAGE CONFIDENCE
        # ====================================================

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

        # ====================================================
        # LOG
        # ====================================================

        logger.info(
            f"RapidOCR: "
            f"{len(boxes)} relevant text regions, "
            f"avg confidence: "
            f"{avg_confidence:.3f}"
        )

        logger.info(
            f"RapidOCR extracted text: "
            f"{full_text[:500]}"
        )

        # ====================================================
        # RETURN
        # ====================================================

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg_confidence,
            engine_used="rapidocr",
        )


# ============================================================
# OCR ENGINE MANAGER
# ============================================================


class OCREngineManager:
    """
    Manages RapidOCR as the single OCR engine.

    The engine is created lazily so OCR models are not loaded
    unless OCR is actually requested.
    """

    def __init__(
        self,
        config: OCRConfig | None = None,
    ) -> None:

        self.config = config or OCRConfig()

        self._engine = None

    # ========================================================
    # GET ENGINE
    # ========================================================

    def _get_engine(
        self,
    ) -> RapidOCREngine:

        if self._engine is None:

            self._engine = (
                RapidOCREngine(
                    self.config
                )
            )

        return self._engine

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """Extract text using RapidOCR."""

        engine = self._get_engine()

        try:

            return engine.extract(
                image
            )

        except (
            RuntimeError,
            ValueError,
            OSError,
        ) as e:

            logger.error(
                f"RapidOCR failed: {e}"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="rapidocr",
            )

    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    def extract_with_both(
        self,
        image: np.ndarray,
    ) -> Tuple[
        OCRResult,
        OCRResult,
    ]:
        """
        Kept for backward compatibility.

        RapidOCR is now the only OCR engine, so both returned
        results are the same OCR result.
        """

        result = self.extract(
            image
        )

        return result, result


# ============================================================
# OCR VISUALIZATION
# ============================================================


def draw_ocr_boxes(
    image: np.ndarray,
    boxes: List[OCRBox],
    color: Tuple[int, int, int] = (
        0,
        255,
        0,
    ),
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
