"""Structured exception hierarchy for pipeline stages."""

from __future__ import annotations


class DocuNetError(Exception):

    def __init__(self, message: str, *, stage: str = "unknown") -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class QualityGateError(DocuNetError):

    def __init__(self, message: str, issues: list[str] | None = None) -> None:
        self.issues = issues or []
        super().__init__(message, stage="quality_gate")


class RectificationError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="rectification")


class EnhancementError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="enhancement")


class GlareRemovalError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="glare_removal")


class TamperDetectionError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="tamper_detection")


class OCRExtractionError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="ocr")


class FieldParsingError(DocuNetError):

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="field_parsing")


class ModelNotFoundError(DocuNetError):

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Model not found: {path}", stage="model_loading")
