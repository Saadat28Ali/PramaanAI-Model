"""OCR sub-package — multi-engine text extraction and field parsing."""

from .ocr_engine import OCREngineManager, OCRResult
from .field_parser import FieldParser, ParsedDocument

__all__ = ["OCREngineManager", "OCRResult", "FieldParser", "ParsedDocument"]
