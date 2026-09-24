"""Lightweight Tesseract OCR text extraction for document processing."""

import cv2
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import shutil
import time

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
    raw_output: Optional[dict] = field(
        default=None,
        repr=False,
    )

    def to_dict(self) -> dict:
        return {
            "boxes": [
                b.to_dict()
                for b in self.boxes
            ],
            "full_text": self.full_text,
            "avg_confidence": round(
                self.avg_confidence,
                4,
            ),
            "engine_used": self.engine_used,
        }


# ============================================================
# BASE OCR ENGINE
# ============================================================


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """Run OCR on the image and return structured results."""
        pass


# ============================================================
# TESSERACT OCR ENGINE
# ============================================================


class TesseractOCREngine(BaseOCREngine):
    """
    Lightweight Tesseract OCR engine for document text extraction.

    Uses the system Tesseract binary.

    Tesseract is loaded lazily so the OCR engine is only
    initialized when OCR is actually requested.
    """

    def __init__(
        self,
        config: OCRConfig | None = None,
    ) -> None:

        self.config = config or OCRConfig()

        self._engine_loaded = False

        # Maximum dimension sent to OCR.
        # Keeping this limited helps control memory and latency.
        self.max_dimension = 1600

        # Ignore extremely low-confidence detections.
        # Tesseract confidence is returned as 0-100.
        self.min_confidence = 20.0

        # Ignore extremely short detections.
        self.min_text_length = 2

        # Tesseract configuration.
        #
        # --oem 1 = LSTM OCR engine
        # --psm 6 = Assume a single uniform block of text
        self.tesseract_config = "--oem 1 --psm 11"

        # Actual Tesseract executable path.
        self.tesseract_cmd = None

    # ========================================================
    # FIND TESSERACT
    # ========================================================

    def _find_tesseract(self) -> str:
        """
        Find the Tesseract executable.

        Windows:
            Uses the standard UB Mannheim installation path.

        Linux/Render:
            Uses PATH lookup.

        Returns:
            Absolute path to tesseract executable.
        """

        # ----------------------------------------------------
        # 1. Environment variable
        # ----------------------------------------------------

        env_path = os.environ.get(
            "TESSERACT_CMD"
        )

        if env_path:

            if os.path.isfile(env_path):

                logger.info(
                    f"Using Tesseract from TESSERACT_CMD: "
                    f"{env_path}"
                )

                return env_path

        # ----------------------------------------------------
        # 2. Windows standard installation
        # ----------------------------------------------------

        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for path in windows_paths:

            if os.path.isfile(path):

                logger.info(
                    f"Using Windows Tesseract: {path}"
                )

                return path

        # ----------------------------------------------------
        # 3. PATH lookup
        # ----------------------------------------------------

        path_result = shutil.which(
            "tesseract"
        )

        if path_result:

            logger.info(
                f"Using Tesseract from PATH: "
                f"{path_result}"
            )

            return path_result

        # ----------------------------------------------------
        # 4. Not found
        # ----------------------------------------------------

        raise FileNotFoundError(
            "Tesseract executable not found. "
            "On Windows install Tesseract to "
            r"C:\Program Files\Tesseract-OCR\ "
            "or set TESSERACT_CMD. "
            "On Linux/Render install the "
            "tesseract-ocr system package."
        )

    # ========================================================
    # LOAD TESSERACT
    # ========================================================

    def _load_engine(self) -> None:
        """Check that the Tesseract binary is available."""

        if self._engine_loaded:
            return

        try:

            import pytesseract

            # Find actual executable.
            self.tesseract_cmd = (
                self._find_tesseract()
            )

            # Explicitly tell pytesseract which
            # executable to use.
            pytesseract.pytesseract.tesseract_cmd = (
                self.tesseract_cmd
            )

            # Verify the executable.
            version = (
                pytesseract.get_tesseract_version()
            )

            logger.info(
                f"Tesseract loaded successfully: "
                f"{version}"
            )

            logger.info(
                f"Tesseract executable: "
                f"{self.tesseract_cmd}"
            )

            self._engine_loaded = True

        except ImportError:

            logger.error(
                "pytesseract is not installed. "
                "Install with: "
                "pip install pytesseract"
            )

            raise

        except Exception as e:

            logger.error(
                f"Tesseract initialization failed: {e}"
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
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
    ) -> List[List[int]]:
        """
        Convert Tesseract bounding box coordinates back to
        the original image coordinate system.

        Returns polygon in the same format used by the
        previous OCR implementation.
        """

        if scale <= 0:
            scale = 1.0

        return [
            [
                int(x / scale),
                int(y / scale),
            ],
            [
                int((x + width) / scale),
                int(y / scale),
            ],
            [
                int((x + width) / scale),
                int((y + height) / scale),
            ],
            [
                int(x / scale),
                int((y + height) / scale),
            ],
        ]

    # ========================================================
    # OCR EXTRACTION
    # ========================================================

    def extract(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """Run Tesseract OCR on the supplied image."""

        # ====================================================
        # START TIMER
        # ====================================================

        ocr_start = time.perf_counter()

        # ====================================================
        # LOAD ENGINE
        # ====================================================

        self._load_engine()

        # ====================================================
        # VALIDATE IMAGE
        # ====================================================

        if image is None or image.size == 0:

            logger.warning(
                "Empty image supplied to Tesseract"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="tesseract",
            )

        # ====================================================
        # IMPORT LAZILY
        # ====================================================

        import pytesseract

        # ====================================================
        # PREPARE IMAGE
        # ====================================================

        image_for_ocr, scale = (
            self._prepare_image(image)
        )

        logger.info(
            f"Tesseract input size: "
            f"{image_for_ocr.shape[1]}x"
            f"{image_for_ocr.shape[0]}"
        )

        # ====================================================
        # RUN OCR
        # ====================================================

        try:

            inference_start = time.perf_counter()

            data = pytesseract.image_to_data(
                image_for_ocr,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            logger.info(
                f"Tesseract inference time: "
                f"{inference_time:.3f}s"
            )

        except Exception as e:

            logger.error(
                f"Tesseract inference failed: {e}"
            )

            raise

        # ====================================================
        # EXTRACT RAW DATA
        # ====================================================

        boxes: List[OCRBox] = []

        texts = data.get(
            "text",
            [],
        )

        confidences = data.get(
            "conf",
            [],
        )

        lefts = data.get(
            "left",
            [],
        )

        tops = data.get(
            "top",
            [],
        )

        widths = data.get(
            "width",
            [],
        )

        heights = data.get(
            "height",
            [],
        )

        # ====================================================
        # CONVERT RESULTS
        # ====================================================

        for (
            text,
            confidence,
            x,
            y,
            width,
            height,
        ) in zip(
            texts,
            confidences,
            lefts,
            tops,
            widths,
            heights,
        ):

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            if len(text) < self.min_text_length:
                continue

            # ------------------------------------------------
            # CONFIDENCE
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

            # Tesseract returns -1 for invalid/empty
            # confidence values.
            if confidence < self.min_confidence:
                continue

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            try:

                bbox_converted = (
                    self._convert_bbox(
                        int(x),
                        int(y),
                        int(width),
                        int(height),
                        scale,
                    )
                )

            except Exception as e:

                logger.warning(
                    f"Could not convert OCR bbox: {e}"
                )

                continue

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            boxes.append(
                OCRBox(
                    text=text,
                    confidence=confidence / 100.0,
                    bbox=bbox_converted,
                    engine="tesseract",
                )
            )

        # ====================================================
        # SORT BOXES
        # ====================================================

        # Top -> bottom
        # Left -> right

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
        # TOTAL OCR TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - ocr_start
        )

        # ====================================================
        # LOG
        # ====================================================

        logger.info(
            f"Tesseract: "
            f"{len(boxes)} relevant text regions, "
            f"avg confidence: "
            f"{avg_confidence:.3f}"
        )

        logger.info(
            f"Tesseract extracted text: "
            f"{full_text[:500]}"
        )

        logger.info(
            f"Tesseract total OCR time: "
            f"{total_time:.3f}s"
        )

        # ====================================================
        # RETURN
        # ====================================================

        return OCRResult(
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg_confidence,
            engine_used="tesseract",
        )


# ============================================================
# OCR ENGINE MANAGER
# ============================================================


class OCREngineManager:
    """
    Manages Tesseract as the single OCR engine.

    The engine is created lazily so the OCR process does not
    consume resources until OCR is actually requested.
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
    ) -> TesseractOCREngine:

        if self._engine is None:

            self._engine = (
                TesseractOCREngine(
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
        """Extract text using Tesseract."""

        engine = self._get_engine()

        try:

            return engine.extract(
                image
            )

        except (
            RuntimeError,
            ValueError,
            OSError,
            FileNotFoundError,
        ) as e:

            logger.error(
                f"Tesseract failed: {e}"
            )

            return OCRResult(
                boxes=[],
                full_text="",
                avg_confidence=0.0,
                engine_used="tesseract",
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

        Tesseract is now the only OCR engine, so both
        returned results are the same OCR result.
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
