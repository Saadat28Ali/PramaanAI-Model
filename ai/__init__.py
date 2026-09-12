"""DocuNet — document verification: preprocessing, forensics, and OCR."""

from .config import DocuNetConfig
from .exceptions import DocuNetError
from .pipeline import DocuNetPipeline, PipelineResult
from .protocols import (
    Enhancer,
    FieldExtractor,
    ForensicAnalyzer,
    GlareRemover,
    OCRExtractor,
    QualityChecker,
    Rectifier,
    Serializable,
    TamperClassifierProtocol,
)

__all__ = [
    "DocuNetConfig",
    "DocuNetError",
    "DocuNetPipeline",
    "PipelineResult",
    "Enhancer",
    "FieldExtractor",
    "ForensicAnalyzer",
    "GlareRemover",
    "OCRExtractor",
    "QualityChecker",
    "Rectifier",
    "Serializable",
    "TamperClassifierProtocol",
]

__version__ = "1.0.0"
__author__ = "DocuNet Team"
