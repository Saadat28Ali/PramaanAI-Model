"""Main pipeline orchestrator chaining all processing stages."""

import cv2
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger


from .config import DocuNetConfig
from .protocols import (
    ForensicAnalyzer,
    QualityChecker,
    Rectifier as RectifierProto,
    Enhancer,
    GlareRemover,
    OCRExtractor,
    FieldExtractor,
)
from .preprocessing.quality_gate import QualityGate, QualityReport
from .preprocessing.rectifier import DocumentRectifier, RectificationResult
from .preprocessing.enhancement import ImageEnhancer
from .preprocessing.glare_handler import GlareHandler, GlareResult
from .forensics.ela_detector import ELADetector, ELAResult
from .forensics.noise_analyzer import NoiseAnalyzer
from .forensics.jpeg_ghost import JPEGGhostDetector
from .forensics.copymove_detector import CopyMoveDetector
from .ocr.ocr_engine import OCREngineManager, OCRResult
from .ocr.field_parser import FieldParser, ParsedDocument


@dataclass
class PipelineResult:
    """Complete result from the DocuNet pipeline."""

    success: bool
    stage_reached: str
    error_message: Optional[str] = None

    quality_report: Optional[QualityReport] = None
    rectification_result: Optional[RectificationResult] = None
    glare_result: Optional[GlareResult] = None
    ela_result: Optional[ELAResult] = None
    dl_tamper_result: Optional[Dict] = None
    noise_analysis_result: Optional[Dict] = None
    jpeg_ghost_result: Optional[Dict] = None
    copymove_result: Optional[Dict] = None
    ocr_result: Optional[OCRResult] = None
    parsed_document: Optional[ParsedDocument] = None
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return sum(self.timings.values())

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            "success": self.success,
            "stage_reached": self.stage_reached,
            "error_message": self.error_message,
            "total_time_ms": round(self.total_time_ms, 2),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }

        if self.quality_report:
            result["quality"] = self.quality_report.to_dict()

        if self.rectification_result:
            result["rectification"] = self.rectification_result.to_dict()

        if self.glare_result:
            result["glare"] = self.glare_result.to_dict()

        if self.ela_result:
            result["tamper_detection"] = self.ela_result.to_dict()

        if self.dl_tamper_result:
            result["dl_tamper"] = self.dl_tamper_result

        if self.noise_analysis_result:
            result["noise_analysis"] = self.noise_analysis_result

        if self.jpeg_ghost_result:
            result["jpeg_ghost"] = self.jpeg_ghost_result

        if self.copymove_result:
            result["copy_move"] = self.copymove_result

        if self.ocr_result:
            result["ocr"] = self.ocr_result.to_dict()

        if self.parsed_document:
            result["document"] = self.parsed_document.to_dict()

        return result


class DocuNetPipeline:
    """Processes a document image through the full verification pipeline."""

    def __init__(self, config: DocuNetConfig | None = None) -> None:
        self.config = config or DocuNetConfig.default()

        self.quality_gate = QualityGate(self.config.quality_gate)
        self.rectifier = DocumentRectifier(self.config.rectification)
        self.enhancer = ImageEnhancer(self.config.enhancement)
        self.glare_handler = GlareHandler(self.config.glare)
        self.ela_detector = ELADetector(self.config.ela)
        self.noise_analyzer = NoiseAnalyzer(
            block_size=self.config.noise_analysis.block_size,
        )
        self.jpeg_ghost_detector = JPEGGhostDetector(
            block_size=self.config.jpeg_ghost.block_size,
            q_low=self.config.jpeg_ghost.q_low,
            q_high=self.config.jpeg_ghost.q_high,
            q_step=self.config.jpeg_ghost.q_step,
        )
        self.copymove_detector = CopyMoveDetector(
            n_features=self.config.copy_move.n_features,
            ratio_thresh=self.config.copy_move.ratio_thresh,
            min_distance=self.config.copy_move.min_distance,
            ransac_reproj=self.config.copy_move.ransac_reproj,
            min_inliers=self.config.copy_move.min_inliers,
        )
        self.ocr_manager = OCREngineManager(self.config.ocr)
        self.field_parser = FieldParser()

        # DL tamper classifier (loaded only if checkpoint exists)
        self.dl_classifier = None
        if os.path.exists(self.config.tamper_model.model_save_path):
            try:
                from src.models.tamper_classifier import TamperClassifierInference
                self.dl_classifier = TamperClassifierInference(
                    self.config.tamper_model.model_save_path,
                    self.config.tamper_model,
                )
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning(f"Could not load DL tamper classifier: {e}")

        logger.info("DocuNet pipeline initialized")

    def process(
        self,
        image: np.ndarray,
        skip_quality_gate: bool = False,
        skip_ocr: bool = False,
    ) -> PipelineResult:
        """Process a single document image through the full pipeline."""
        result = PipelineResult(success=False, stage_reached="input")
        result.images["original"] = image.copy()

        if not skip_quality_gate:
            t0 = time.time()
            quality_report = self.quality_gate.evaluate(image)
            result.timings["quality_gate"] = (time.time() - t0) * 1000
            result.quality_report = quality_report
            result.stage_reached = "quality_gate"

            if not quality_report.passed:
                result.error_message = "; ".join(quality_report.issues)
                logger.warning(
                    f"Quality gate flagged issues (continuing): {result.error_message}"
                )

        t0 = time.time()
        rectification = self.rectifier.rectify(image)
        result.timings["rectification"] = (time.time() - t0) * 1000
        result.rectification_result = rectification
        result.stage_reached = "rectification"

        if rectification.success:
            working_image = rectification.rectified
            result.images["rectified"] = working_image.copy()

            if rectification.corners is not None:
                result.images["corners_vis"] = self.rectifier.draw_corners(
                    image, rectification.corners
                )
        else:
            working_image = image.copy()
            logger.info("Rectification failed; proceeding with original image")

        t0 = time.time()
        glare_result = self.glare_handler.process(working_image)
        result.timings["glare_handling"] = (time.time() - t0) * 1000
        result.glare_result = glare_result
        result.stage_reached = "glare_handling"

        if glare_result.has_glare and glare_result.recoverable:
            working_image = glare_result.inpainted
            result.images["glare_removed"] = working_image.copy()
            result.images["glare_mask"] = glare_result.glare_mask

        t0 = time.time()
        enhanced = self.enhancer.enhance(working_image)
        result.timings["enhancement"] = (time.time() - t0) * 1000
        result.images["enhanced"] = enhanced.copy()
        result.stage_reached = "enhancement"

        t0 = time.time()
        ela_result = self.ela_detector.analyze(working_image)
        result.timings["ela_detection"] = (time.time() - t0) * 1000
        result.ela_result = ela_result
        result.stage_reached = "tamper_detection"
        result.images["ela_heatmap"] = ela_result.heatmap
        result.images["ela_overlay"] = self.ela_detector.overlay_heatmap(
            working_image, ela_result.heatmap
        )

        if self.dl_classifier:
            t0 = time.time()
            dl_result = self.dl_classifier.classify(working_image)
            result.timings["dl_tamper"] = (time.time() - t0) * 1000
            result.dl_tamper_result = dl_result

        t0 = time.time()
        noise_result = self.noise_analyzer.analyze(working_image)
        result.timings["noise_analysis"] = (time.time() - t0) * 1000
        result.noise_analysis_result = noise_result.to_dict()
        result.images["noise_heatmap"] = self.noise_analyzer.visualize(
            noise_result.block_variance_map, working_image.shape[:2]
        )

        t0 = time.time()
        ghost_result = self.jpeg_ghost_detector.analyze(working_image)
        result.timings["jpeg_ghost"] = (time.time() - t0) * 1000
        result.jpeg_ghost_result = ghost_result.to_dict()
        result.images["jpeg_ghost_heatmap"] = ghost_result.ghost_heatmap

        t0 = time.time()
        copymove_result = self.copymove_detector.detect(working_image)
        result.timings["copy_move"] = (time.time() - t0) * 1000
        result.copymove_result = copymove_result.to_dict()
        if copymove_result.is_copymove:
            result.images["copymove_vis"] = copymove_result.visualisation

        if not skip_ocr:
            t0 = time.time()
            try:
                ocr_result = self.ocr_manager.extract(enhanced)
                result.timings["ocr"] = (time.time() - t0) * 1000
                result.ocr_result = ocr_result
                result.stage_reached = "ocr"

                t0 = time.time()
                parsed = self.field_parser.parse(ocr_result)
                result.timings["field_parsing"] = (time.time() - t0) * 1000
                result.parsed_document = parsed
                result.stage_reached = "field_parsing"

            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"OCR failed: {e}")
                result.error_message = f"OCR extraction failed: {str(e)}"
                # Continue — tamper detection results are still valid

        result.success = True
        result.stage_reached = "complete"

        ocr_conf = f"{result.ocr_result.avg_confidence:.3f}" if result.ocr_result else "N/A"
        logger.info(
            f"Pipeline completed in {result.total_time_ms:.1f}ms | "
            f"Tamper score: {ela_result.anomaly_score:.3f} | "
            f"OCR confidence: {ocr_conf}"
        )

        return result

    def process_file(
        self,
        image_path: str,
        skip_quality_gate: bool = False,
        skip_ocr: bool = False,
    ) -> PipelineResult:
        """Load an image from disk and process it."""
        image = cv2.imread(image_path)
        if image is None:
            return PipelineResult(
                success=False,
                stage_reached="input",
                error_message=f"Could not read image: {image_path}",
            )
        return self.process(image, skip_quality_gate, skip_ocr)

    def save_results(
        self,
        result: PipelineResult,
        output_dir: str,
        prefix: str = "docunet",
    ) -> Dict[str, str]:
        """Save all result images and the JSON report to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved = {}

        for name, img in result.images.items():
            if img is not None and len(img.shape) >= 2:
                filepath = output_path / f"{prefix}_{name}.jpg"
                cv2.imwrite(str(filepath), img)
                saved[name] = str(filepath)

        import json
        report_path = output_path / f"{prefix}_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        saved["report"] = str(report_path)

        logger.info(f"Results saved to {output_dir}")
        return saved
