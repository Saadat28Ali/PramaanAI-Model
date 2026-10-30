"""Main pipeline orchestrator chaining all processing stages."""

import cv2
import numpy as np
import time
import os
import base64
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger


from ai.config import DocuNetConfig
from ai.protocols import (
    ForensicAnalyzer,
    QualityChecker,
    Rectifier as RectifierProto,
    Enhancer,
    GlareRemover,
    OCRExtractor,
    FieldExtractor,
)

from ai.preprocessing.quality_gate import QualityGate, QualityReport
from ai.preprocessing.rectifier import DocumentRectifier, RectificationResult
from ai.preprocessing.enhancement import ImageEnhancer
from ai.preprocessing.glare_handler import GlareHandler, GlareResult
from ai.forensics.ela_detector import ELADetector, ELAResult
from ai.forensics.noise_analyzer import NoiseAnalyzer
from ai.forensics.jpeg_ghost import JPEGGhostDetector
from ai.forensics.copymove_detector import CopyMoveDetector
from ai.ocr.ocr_engine import OCREngineManager, OCRResult
from ai.ocr.field_parser import FieldParser, ParsedDocument
from ai.face.verifier import FaceVerifier


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
    face_verification_result: Optional[Dict] = None
    usable: Optional[bool] = None
    usability_status: Optional[str] = None
    student_message: Optional[str] = None
    reupload_reasons: List[str] = field(default_factory=list)
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return sum(self.timings.values())

    def _get_final_tamper_decision(self) -> tuple[bool, List[str]]:
        """Combine ELA, Copy-Move, and DL tamper detection."""

        reasons: List[str] = []
        is_tampered = False

        # ELA
        if self.ela_result is not None:
            if bool(self.ela_result.is_tampered):
                is_tampered = True
                reasons.append("ELA tamper detection")

        # Copy-Move
        if self.copymove_result:
            if bool(self.copymove_result.get("is_copymove", False)):
                is_tampered = True
                reasons.append("Copy-move detection")

        # Deep-learning tamper classifier
        if self.dl_tamper_result:
            if isinstance(self.dl_tamper_result, dict):

                if "is_tampered" in self.dl_tamper_result:
                    dl_flag = self.dl_tamper_result.get("is_tampered")

                elif "tampered" in self.dl_tamper_result:
                    dl_flag = self.dl_tamper_result.get("tampered")

                else:
                    dl_flag = False

                if dl_flag is True:
                    is_tampered = True
                    reasons.append(
                        "Deep-learning tamper detection"
                    )

        return is_tampered, reasons

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            "success": self.success,
            "stage_reached": self.stage_reached,
            "error_message": self.error_message,
            "total_time_ms": round(self.total_time_ms, 2),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }

        # Student-facing usability decision is populated by process().
        if self.usable is not None:
            result["usable"] = self.usable
            result["status"] = self.usability_status
            result["message"] = self.student_message
            result["reupload_reasons"] = list(self.reupload_reasons)

        if self.quality_report:
            result["quality"] = self.quality_report.to_dict()

        if self.rectification_result:
            result["rectification"] = self.rectification_result.to_dict()

        if self.glare_result:
            result["glare"] = self.glare_result.to_dict()

        if self.ela_result:
            tamper_result = self.ela_result.to_dict()

            final_is_tampered, reasons = (
                self._get_final_tamper_decision()
            )

            tamper_result["is_tampered"] = final_is_tampered

            if final_is_tampered:
                tamper_result["message"] = (
                    "Tampering detected by forensic analysis."
                )
            else:
                tamper_result["message"] = (
                    f"No tampering detected "
                    f"(score: "
                    f"{self.ela_result.anomaly_score:.3f})."
                )

            tamper_result["detection_reasons"] = reasons

            result["tamper_detection"] = tamper_result

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

        if self.face_verification_result is not None:
            result["face_verification"] = self.face_verification_result

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

        # Face verifier is initialized only when selfie verification is requested.
        self.face_verifier = None

        # DL tamper classifier (loaded only if checkpoint exists)
        self.dl_classifier = None
        if os.path.exists(self.config.tamper_model.model_save_path):
            try:
                from ai.models.tamper_classifier import TamperClassifierInference
                self.dl_classifier = TamperClassifierInference(
                    self.config.tamper_model.model_save_path,
                    self.config.tamper_model,
                )
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning(f"Could not load DL tamper classifier: {e}")

        logger.info("DocuNet pipeline initialized")

    def process_document_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Student-side workflow.

        Evaluates the ORIGINAL upload for usability, then runs document
        rectification, glare handling, and enhancement. Forensic analysis is
        deliberately not run here.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "success": False,
                "usable": False,
                "status": "INVALID_IMAGE",
                "error_message": "A valid image is required.",
            }

        timings: Dict[str, float] = {}
        recommendations: List[str] = []

        # Quality must be evaluated on the untouched upload.
        t0 = time.time()
        quality_report = self.quality_gate.evaluate(image)
        timings["quality_gate"] = (time.time() - t0) * 1000

        # Reject empty/near-black frames independently of the other metrics.
        is_black_frame = self.quality_gate.check_black_frame(image)
        quality_data = quality_report.to_dict()

        if is_black_frame:
            recommendations.append(
                "The image appears too dark or blank. Capture the document again."
            )

        recommendations.extend(quality_report.issues)

        # Rectification is useful preprocessing, but failure to find a border
        # is not automatically a rejection: some uploads are already tightly
        # cropped around the document.
        t0 = time.time()
        rectification = self.rectifier.rectify(image)
        timings["rectification"] = (time.time() - t0) * 1000
        working_image = (
            rectification.rectified.copy()
            if rectification.success and rectification.rectified is not None
            else image.copy()
        )

        t0 = time.time()
        glare_result = self.glare_handler.process(working_image)
        timings["glare_handling"] = (time.time() - t0) * 1000

        # Inpainting may still be useful for internal preprocessing, but it
        # does not make a glare-affected upload acceptable for student submission.
        if glare_result.has_glare and glare_result.recoverable:
            working_image = glare_result.inpainted.copy()

        t0 = time.time()
        enhanced_image = self.enhancer.enhance(working_image)
        timings["enhancement"] = (time.time() - t0) * 1000

        # Student-side policy: reject an upload whenever the glare detector
        # reports glare, regardless of whether the glare is recoverable.
        # Recoverability describes preprocessing, not submission acceptability.
        glare_detected = bool(glare_result.has_glare)

        reupload_reasons: List[str] = []
        if not quality_report.passed:
            reupload_reasons.append("QUALITY_CHECK_FAILED")
        if is_black_frame:
            reupload_reasons.append("BLACK_OR_EMPTY_IMAGE")
        if glare_detected:
            reupload_reasons.append("GLARE_DETECTED")

        usable = len(reupload_reasons) == 0

        if not rectification.success:
            recommendations.append(
                "Document boundaries were not detected; check that the full document is visible."
            )

        if glare_detected:
            recommendations.append(
                "Glare or reflection was detected on your document. Please retake "
                "the photo in even lighting, avoid direct flash and reflections, "
                "and make sure all text and details are clearly visible."
            )

        if not quality_report.passed:
            recommendations.append(
                "The document did not pass the image-quality checks. Please upload "
                "a clearer image with the full document visible."
            )

        if is_black_frame:
            recommendations.append(
                "The uploaded image appears blank or too dark. Please capture the "
                "document again in adequate lighting."
            )

        if usable:
            student_message = (
                "Your document passed the image-quality checks and is ready for "
                "the next verification stage."
            )
        elif glare_detected:
            student_message = (
                "Glare or reflection was detected on your document. Please upload "
                "a new image captured in even lighting, without direct flash or "
                "reflections. Ensure all text and details are clearly visible."
            )
        else:
            student_message = (
                "Your document could not be accepted. Please upload a clearer "
                "image and follow the capture instructions."
            )

        return {
            "success": True,
            "workflow": "student_document_quality",
            "usable": usable,
            "status": "USABLE" if usable else "REUPLOAD_REQUIRED",
            "message": student_message,
            "reupload_reasons": reupload_reasons,
            "checks": {
                "quality_gate": quality_data,
                "black_frame": {
                    "passed": not is_black_frame,
                    "detected": is_black_frame,
                },
                "rectification": rectification.to_dict(),
                "glare": {
                    **glare_result.to_dict(),
                    "has_glare": glare_detected,
                    "recoverable": bool(glare_result.recoverable),
                    "measurement": (
                        "Filtered glare mask after connected-component filtering "
                        "and dilation; not directly comparable to the quality-gate "
                        "high-intensity pixel ratio."
                    ),
                    "severity": (
                        "severe" if glare_detected and not glare_result.recoverable
                        else "detected" if glare_detected
                        else "none"
                    ),
                    "blocks_usability": glare_detected,
                },
                "enhancement": {
                    "applied": True,
                    "output_width": int(enhanced_image.shape[1]),
                    "output_height": int(enhanced_image.shape[0]),
                },
            },
            "recommendations": list(dict.fromkeys(recommendations)),
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "total_time_ms": round(sum(timings.values()), 2),
        }

    def process_tamper_analysis(
        self, image: np.ndarray, include_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Officer-side workflow.

        Runs forensic detectors on the ORIGINAL uploaded image. Do not run
        CLAHE, denoising, sharpening, inpainting, or perspective correction
        before forensic analysis because those operations alter image evidence.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "success": False,
                "analysis_completed": False,
                "error_message": "A valid image is required.",
            }

        timings: Dict[str, float] = {}
        checks: Dict[str, Any] = {}
        # ELA
        t0 = time.time()
        ela_result = self.ela_detector.analyze(image)
        timings["ela"] = (time.time() - t0) * 1000
        ela_data = ela_result.to_dict()
        checks["ela"] = {
            **ela_data,
            "suspicious": bool(ela_result.is_tampered),
        }
        # Noise inconsistency. Use the same configured threshold for both
        # the decision and the user-facing message (the detector's own message
        # may use a different hard-coded threshold).
        t0 = time.time()
        noise_result = self.noise_analyzer.analyze(image)
        timings["noise_analysis"] = (time.time() - t0) * 1000
        noise_data = noise_result.to_dict()
        noise_threshold = float(
            self.config.noise_analysis.inconsistency_threshold
        )
        noise_score = float(noise_result.inconsistency_score)
        noise_suspicious = bool(noise_score > noise_threshold)
        noise_message = (
            "Noise variation exceeds the configured threshold. "
            "Localized differences in image noise were identified. "
            "This can occur due to editing or image processing; review "
            "the highlighted regions alongside the other forensic findings."
            if noise_suspicious
            else
            "Noise variation is within the configured threshold. "
            "This detector did not identify a significant noise inconsistency."
        )
        checks["noise_analysis"] = {
            **noise_data,
            "inconsistency_score": noise_score,
            "threshold": noise_threshold,
            "suspicious": noise_suspicious,
            "message": noise_message,
        }

        # JPEG Ghost. Keep global score and local block outliers separate.
        # 0.30 is the threshold already used by JPEGGhostDetector's own
        # interpretation; this pipeline does not alter the detector itself.
        t0 = time.time()
        ghost_result = self.jpeg_ghost_detector.analyze(image)
        timings["jpeg_ghost"] = (time.time() - t0) * 1000
        ghost_data = ghost_result.to_dict()
        ghost_score = float(ghost_result.inconsistency_score)
        ghost_global_threshold = 0.30
        ghost_global_anomaly = bool(
            ghost_score > ghost_global_threshold
        )
        ghost_local_outliers = bool(ghost_result.suspicious_blocks)
        ghost_message = (
            "Global JPEG compression inconsistency exceeds the detector's "
            "reference threshold. Local block findings should be reviewed."
            if ghost_global_anomaly and ghost_local_outliers
            else
            "Global JPEG compression inconsistency exceeds the detector's "
            "reference threshold. No local outlier blocks were reported."
            if ghost_global_anomaly
            else
            "Global JPEG compression appears consistent with the detector's "
            "reference threshold, but localized compression outlier blocks "
            "were found. These can also arise from image content or normal "
            "processing and are not proof of editing."
            if ghost_local_outliers
            else
            "Global JPEG compression appears consistent with the detector's "
            "reference threshold, and no local outlier blocks were reported."
        )
        checks["jpeg_ghost"] = {
            **ghost_data,
            "inconsistency_score": ghost_score,
            "global_threshold": ghost_global_threshold,
            "global_anomaly_detected": ghost_global_anomaly,
            "local_outlier_blocks_detected": ghost_local_outliers,
            # Preserve the existing aggregate field for API compatibility.
            "suspicious": bool(
                ghost_global_anomaly or ghost_local_outliers
            ),
            "message": ghost_message,
            "decision_note": (
                "Global inconsistency and local block outliers are separate "
                "signals. Neither is a calibrated probability of fraud."
            ),
        }

        # Copy-move. The detector's confidence is a heuristic score, not a
        # calibrated probability that the document is fraudulent.
        t0 = time.time()
        copymove_result = self.copymove_detector.detect(image)
        timings["copy_move"] = (time.time() - t0) * 1000
        copymove_data = copymove_result.to_dict()
        copymove_flag = bool(copymove_result.is_copymove)
        copymove_matches = int(
            getattr(copymove_result, "match_count", 0) or 0
        )
        copymove_clusters = len(
            getattr(copymove_result, "matched_regions", []) or []
        )
        copymove_confidence = getattr(
            copymove_result, "confidence", None
        )
        if copymove_flag:
            copymove_message = (
                f"Potential duplicated image content was identified "
                f"({copymove_matches} geometrically consistent matches"
                + (
                    f", {copymove_clusters} region cluster(s)"
                    if copymove_clusters else ""
                )
                + "). Repeated natural patterns can also produce matches; "
                "manual inspection is recommended."
            )
        else:
            copymove_message = (
                "No qualifying copy-move pattern was detected by this check. "
                "This does not rule out other forms of manipulation."
            )
        checks["copy_move"] = {
            **copymove_data,
            "is_copymove": copymove_flag,
            "suspicious": copymove_flag,
            "message": copymove_message,
            "confidence_interpretation": (
                "Heuristic detector score; not a calibrated fraud probability."
            ),
        }
        if copymove_confidence is not None:
            checks["copy_move"]["confidence"] = float(copymove_confidence)
        # Optional deep-learning classifier
        if self.dl_classifier is None:
            checks["deep_learning"] = {
                "available": False,
                "suspicious": None,
                "message": "Tamper classifier checkpoint is not configured or could not be loaded.",
            }
        else:
            t0 = time.time()
            try:
                dl_result = self.dl_classifier.classify(image)
                timings["deep_learning"] = (time.time() - t0) * 1000
                if isinstance(dl_result, dict):
                    dl_flag = dl_result.get(
                        "is_tampered", dl_result.get("tampered")
                    )
                    if dl_flag is None:
                        label = str(dl_result.get("label", "")).strip().lower()
                        prediction = dl_result.get("prediction")
                        if label == "tampered":
                            dl_flag = True
                        elif label in {"authentic", "real", "genuine"}:
                            dl_flag = False
                        elif prediction in (0, 1):
                            dl_flag = bool(prediction == 1)

                    checks["deep_learning"] = {
                        **dl_result,
                        "available": True,
                        "suspicious": bool(dl_flag) if dl_flag is not None else None,
                    }
                else:
                    checks["deep_learning"] = {
                        "available": True,
                        "suspicious": None,
                        "result": str(dl_result),
                        "message": "Classifier returned an unsupported result format.",
                    }
            except Exception as exc:
                logger.exception("Deep-learning tamper inference failed")
                timings["deep_learning"] = (time.time() - t0) * 1000
                checks["deep_learning"] = {
                    "available": True,
                    "suspicious": None,
                    "error": str(exc),
                }

        # Only explicit positive detector flags contribute to this combined
        # indicator. Missing/unavailable checks are not treated as negatives.
        positive_checks = [
            name for name, data in checks.items()
            if data.get("suspicious") is True
        ]
        completed_checks = [
            name for name, data in checks.items()
            if data.get("suspicious") is not None
        ]

        if positive_checks:
            overall_status = "SUSPICIOUS_INDICATORS_FOUND"
            overall_message = (
                "Potential forensic anomalies were identified. These findings "
                "are indicators for further review, not confirmation that the "
                "document is fraudulent or has been tampered with."
            )
            recommendation = (
                "Manually inspect the highlighted regions and compare the "
                "findings with the original document or a trusted source before "
                "making a verification decision."
            )
        else:
            overall_status = "NO_SUSPICIOUS_INDICATORS_DETECTED"
            overall_message = (
                "No suspicious indicators were identified by the available "
                "forensic checks. This result does not guarantee that the "
                "document is authentic."
            )
            recommendation = (
                "Continue with the remaining identity and document "
                "verification steps; do not treat this result alone as proof "
                "of authenticity."
            )

        response = {
            "success": True,
            "workflow": "officer_tamper_analysis",
            "analysis_completed": True,
            # Backward-compatible legacy field. It represents positive
            # detector indicators, not a confirmed fraud verdict.
            "tampering_detected": bool(positive_checks),
            "suspicious_indicators_found": bool(positive_checks),
            "overall_status": overall_status,
            "overall_message": overall_message,
            "recommendation": recommendation,
            "checks": checks,
            "detection_reasons": positive_checks,
            "checks_run": list(checks.keys()),
            "checks_with_decisions": completed_checks,
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "total_time_ms": round(sum(timings.values()), 2),
        }

        # Visualization is opt-in. Return the original document image without
        # detector bounding boxes, labels, colored regions, or a legend.
        # Keep the existing Base64 response key so the current Postman visualizer
        # and API clients remain compatible.
        if include_visualization:
            success, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if success:
                image_base64 = base64.b64encode(
                    encoded.tobytes()
                ).decode("ascii")
                response["visualization"] = {
                    "forensic_overlay_base64": image_base64,
                    "image_format": "image/jpeg",
                    "annotations": False,
                    "description": "Original document image without bounding boxes or detector annotations.",
                }
            else:
                response["visualization"] = {
                    "available": False,
                    "message": "Could not encode the document image.",
                }

        return response

    def process(
        self,
        image: np.ndarray,
        skip_quality_gate: bool = False,
        skip_ocr: bool = False,
        selfie_image: Optional[np.ndarray] = None,
        doc_type: Optional[str] = None,
    ) -> PipelineResult:
        """Process a document image through the full pipeline.

        If selfie_image is provided, face verification is performed
        using the rectified document image and the selfie.
        """
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

        # Face verification is optional. Existing document-only behavior
        # remains unchanged when selfie_image is not provided.
        if selfie_image is not None:
            t0 = time.time()
            try:
                # Initialize the face verifier only when it is actually needed.
                if self.face_verifier is None:
                    self.face_verifier = FaceVerifier()

                face_result = self.face_verifier.compare_document_and_selfie(
                    document_image=working_image,
                    selfie_image=selfie_image,
                )
                result.face_verification_result = face_result
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Face verification failed: {e}")
                result.face_verification_result = {
                    "similarity_score": None,
                    "threshold": 0.45,
                    "match": False,
                    "error": str(e),
                }

            result.timings["face_verification"] = (time.time() - t0) * 1000

        t0 = time.time()
        glare_result = self.glare_handler.process(working_image)
        result.timings["glare_handling"] = (time.time() - t0) * 1000
        result.glare_result = glare_result
        result.stage_reached = "glare_handling"

        # Student verification policy: any glare detected by GlareHandler
        # requires a fresh upload, even if preprocessing can recover the region.
        # Recoverability is a preprocessing property, not an acceptance criterion.
        reupload_reasons: List[str] = []
        if result.quality_report is not None and not result.quality_report.passed:
            reupload_reasons.append("QUALITY_CHECK_FAILED")
        if glare_result.has_glare:
            reupload_reasons.append("GLARE_DETECTED")

        result.usable = len(reupload_reasons) == 0
        result.usability_status = (
            "USABLE" if result.usable else "REUPLOAD_REQUIRED"
        )
        result.reupload_reasons = reupload_reasons

        if result.usable:
            result.student_message = (
                "Your document passed the image-quality checks and is ready "
                "for the next verification stage."
            )
        elif glare_result.has_glare:
            result.student_message = (
                "Glare or reflection was detected on your document. Please "
                "upload a new image captured in even lighting, without direct "
                "flash or reflections. Ensure all text and details are clearly visible."
            )
        else:
            result.student_message = (
                "Your document did not pass the image-quality checks. Please "
                "upload a clearer image with the full document visible."
            )

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

        # OCR extraction and field parsing are intentionally disabled.
        # All other pipeline stages continue to run unchanged.
        logger.info("OCR and field parsing are disabled.")

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
        selfie_image: Optional[np.ndarray] = None,
        doc_type: Optional[str] = None,
    ) -> PipelineResult:
        """Load an image from disk and process it."""
        image = cv2.imread(image_path)
        if image is None:
            return PipelineResult(
                success=False,
                stage_reached="input",
                error_message=f"Could not read image: {image_path}",
            )

        return self.process(
            image=image,
            skip_quality_gate=skip_quality_gate,
            skip_ocr=skip_ocr,
            selfie_image=selfie_image,
            doc_type=doc_type,
        )

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
