"""Structural sub-typing contracts (PEP 544) for the DocuNet pipeline."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Serializable(Protocol):
    """Any object that can convert itself to a JSON-friendly dict."""

    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class QualityChecker(Protocol):
    """Evaluates image quality and returns a serialisable report."""

    def evaluate(self, image: np.ndarray) -> Serializable: ...


@runtime_checkable
class Rectifier(Protocol):
    """Detects document boundaries and applies perspective correction."""

    def rectify(self, image: np.ndarray) -> Serializable: ...


@runtime_checkable
class Enhancer(Protocol):
    """Applies document-specific image enhancement for OCR."""

    def enhance(self, image: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class GlareRemover(Protocol):
    """Detects and inpaints glare regions on laminated documents."""

    def process(self, image: np.ndarray) -> Serializable: ...


@runtime_checkable
class ForensicAnalyzer(Protocol):
    """Any module that performs pixel-level forensic analysis."""

    def analyze(self, image: np.ndarray) -> Serializable: ...


@runtime_checkable
class TamperClassifierProtocol(Protocol):
    """DL-based tamper classifier (inference only)."""

    def classify(self, image: np.ndarray) -> Dict[str, Any]: ...


@runtime_checkable
class OCRExtractor(Protocol):
    """Extracts text and bounding boxes from an image."""

    def extract(self, image: np.ndarray) -> Serializable: ...


@runtime_checkable
class FieldExtractor(Protocol):
    """Parses structured fields (Aadhaar, PAN, dates) from raw OCR."""

    def parse(self, ocr_result: Any) -> Serializable: ...
