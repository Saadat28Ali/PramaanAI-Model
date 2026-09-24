"""Extract structured fields from raw OCR text."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .ocr_engine import OCRBox, OCRResult


@dataclass
class DocumentField:
    """A single extracted field from the document."""
    field_name: str
    value: str
    confidence: float
    source_box: Optional[OCRBox] = None


@dataclass
class ParsedDocument:
    """Complete parsed document with structured fields."""
    fields: Dict[str, DocumentField]
    document_type: str
    overall_confidence: float
    raw_text: str
    unmatched_text: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": {
                k: {
                    "value": v.value,
                    "confidence": round(v.confidence, 4),
                }
                for k, v in self.fields.items()
            },
            "document_type": self.document_type,
            "overall_confidence": round(self.overall_confidence, 4),
            "raw_text": self.raw_text,
            "unmatched_text": self.unmatched_text,
        }


# Indian Aadhaar Card
AADHAAR_PATTERNS = {
    "aadhaar_number": r"(?<![/\-\.])\b(\d{4}\s*\d{4}\s*\d{4})\b",
    "dob": r"\b(?:DOB|Date of Birth|Year of Birth)[:\s]*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b",
    "gender": r"\b(MALE|FEMALE|male|female|Male|Female)\b",
    "vid": r"\b(?:VID)[:\s]*(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b",
}


# Indian PAN Card
PAN_PATTERNS = {
    "pan_number": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "dob": r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b",
}


# US Driver's License (generic patterns)
DRIVERS_LICENSE_PATTERNS = {
    "license_number": r"\b(?:DL|LIC|LICENSE)[:\s#]*([A-Z0-9]{6,12})\b",
    "dob": r"\b(?:DOB|DATE OF BIRTH)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})\b",
    "expiry": r"\b(?:EXP|EXPIRES?)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})\b",
    "class": r"\b(?:CLASS)[:\s]*([A-Z])\b",
}


# Generic patterns that work across document types
GENERIC_PATTERNS = {
    "name": r"(?:Name|NAME)[:\s]*([A-Za-z\s\.]+?)(?:\n|$)",
    "dob": r"(?:DOB|Date of Birth|D\.O\.B|Birth)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    "address": r"(?:Address|ADDR|ADD)[:\s]*([\w\s,\.\-#]+?)(?:\n\n|$)",
    "phone": r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "date_generic": r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
}


# Keywords to identify document type
DOC_TYPE_KEYWORDS = {
    "aadhaar": [
        "aadhaar",
        "uidai",
        "unique identification",
        "आधार",
    ],
    "pan": [
        "income tax",
        "permanent account",
        "pan card",
        "govt of india",
    ],
    "drivers_license": [
        "driver",
        "license",
        "driving",
        "motor vehicle",
        "dmv",
    ],
    "passport": [
        "passport",
        "republic of india",
        "nationality",
    ],
}


class FieldParser:
    """Extracts structured fields from raw OCR output."""

    def __init__(self) -> None:
        self.pattern_sets = {
            "aadhaar": AADHAAR_PATTERNS,
            "pan": PAN_PATTERNS,
            "drivers_license": DRIVERS_LICENSE_PATTERNS,
        }

    def parse(self, ocr_result: OCRResult) -> ParsedDocument:
        """Parse OCR results into structured document fields."""

        full_text = ocr_result.full_text
        boxes = ocr_result.boxes

        doc_type = self._detect_document_type(full_text, boxes)

        logger.info(
            f"Detected document type: {doc_type}"
        )

        fields = {}

        if doc_type in self.pattern_sets:
            type_fields = self._extract_with_patterns(
                full_text,
                self.pattern_sets[doc_type],
                boxes,
            )

            fields.update(type_fields)

        generic_fields = self._extract_with_patterns(
            full_text,
            GENERIC_PATTERNS,
            boxes,
        )

        for key, value in generic_fields.items():
            if key not in fields:
                fields[key] = value

        # Aadhaar number can be split across OCR boxes, so recover it
        # independently of the full-text regex.
        if doc_type == "aadhaar" and "aadhaar_number" not in fields:
            aadhaar_field = self._detect_aadhaar_number_from_boxes(boxes)
            if aadhaar_field:
                fields["aadhaar_number"] = aadhaar_field

        if "name" not in fields:
            name_field = self._extract_name_spatial(boxes)

            if name_field:
                fields["name"] = name_field

        matched_texts = {
            f.value.strip().lower()
            for f in fields.values()
        }

        unmatched = [
            b.text
            for b in boxes
            if (
                b.text.strip().lower() not in matched_texts
                and len(b.text.strip()) > 2
            )
        ]

        if fields:
            overall_conf = (
                sum(f.confidence for f in fields.values())
                / len(fields)
            )
        else:
            overall_conf = 0.0

        return ParsedDocument(
            fields=fields,
            document_type=doc_type,
            overall_confidence=overall_conf,
            raw_text=full_text,
            unmatched_text=unmatched,
        )

    def _detect_document_type(self, text: str, boxes: Optional[List[OCRBox]] = None) -> str:
        """
        Detect document type using OCR keywords plus structural evidence.

        Important for Tesseract:
        AADHAAR is often read incorrectly (for example as fragments such as
        "ERNMEN IND"), so the literal word AADHAAR is not required.
        """
        boxes = boxes or []
        text_lower = text.lower()

        scores = {
            "aadhaar": 0,
            "pan": 0,
            "drivers_license": 0,
            "passport": 0,
        }

        # Normal keyword matching.
        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[doc_type] += 1

        # Strong Aadhaar-specific evidence.
        scores["aadhaar"] += self._aadhaar_evidence(text, boxes)

        # PAN-specific detection.
        if re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text.upper()):
            scores["pan"] += 4

        # Passport-specific detection.
        if "passport" in text_lower:
            scores["passport"] += 4

        # Driving-license detection.
        if re.search(
            r"\b(driver|driving|license|licence|motor vehicle|dmv)\b",
            text_lower,
        ):
            scores["drivers_license"] += 2

        valid_scores = {
            doc_type: score
            for doc_type, score in scores.items()
            if score > 0
        }

        if not valid_scores:
            return "unknown"

        # Aadhaar needs enough combined evidence to avoid false positives.
        # Other document types retain the original lightweight behavior.
        if scores["aadhaar"] >= 5:
            return "aadhaar"

        # If Aadhaar evidence is weak, don't let a single weak signal classify it.
        strongest = max(valid_scores.values())
        candidates = [
            doc_type for doc_type, score in valid_scores.items()
            if score == strongest
        ]

        if len(candidates) == 1:
            return candidates[0]

        return "unknown"

    def _aadhaar_evidence(self, text: str, boxes: List[OCRBox]) -> int:
        """
        Build lightweight Aadhaar evidence from OCR text + OCR boxes.

        Tesseract may miss or garble the word AADHAAR, so detection uses
        multiple independent signals instead of depending on that keyword.
        """
        score = 0
        text_lower = text.lower()

        # Strong explicit signals.
        if "aadhaar" in text_lower:
            score += 5
        if "uidai" in text_lower:
            score += 5
        if "unique identification" in text_lower:
            score += 4

        # OCR-tolerant Aadhaar header fragments.
        if re.search(r"\baadh[a-z_]*\b", text_lower):
            score += 2

        # Clean 12-digit number in OCR text.
        if re.search(r"(?<!\d)\d{12}(?!\d)", re.sub(r"\D", "", text)):
            score += 5

        # Look for split numeric OCR boxes which together form 12 digits.
        numeric_groups = []
        for box in boxes:
            digits = re.sub(r"\D", "", box.text)
            if 2 <= len(digits) <= 5:
                numeric_groups.append((box, digits))

        for i in range(len(numeric_groups)):
            combined = ""
            for j in range(i, min(i + 4, len(numeric_groups))):
                combined += numeric_groups[j][1]
                if len(combined) == 12:
                    score += 5
                    break
                if len(combined) > 12:
                    break

        # Supporting signals only.
        if re.search(r"\b(?:male|female)\b", text_lower):
            score += 1

        if re.search(
            r"\b(?:dob|date of birth|year of birth)\b|"
            r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}\b",
            text_lower,
        ):
            score += 1

        indian_terms = (
            "india", "delhi", "mumbai", "gurgaon", "gurugram",
            "noida", "bangalore", "bengaluru", "pune", "jaipur",
            "uttar pradesh", "haryana", "rajasthan", "maharashtra"
        )
        if any(term in text_lower for term in indian_terms):
            score += 1

        return score

    def _detect_aadhaar_number_from_boxes(
        self,
        boxes: List[OCRBox],
    ) -> Optional[DocumentField]:
        """
        Recover a likely Aadhaar number when Tesseract splits it into boxes.
        Only exactly 12 numeric digits are accepted.
        """
        if not boxes:
            return None

        numeric = []
        for box in boxes:
            digits = re.sub(r"\D", "", box.text.strip())

            if 2 <= len(digits) <= 5:
                numeric.append((box, digits))

        for i in range(len(numeric)):
            parts = []
            confidences = []

            for j in range(i, min(i + 4, len(numeric))):
                parts.append(numeric[j][1])
                confidences.append(numeric[j][0].confidence)

                candidate = "".join(parts)

                if len(candidate) == 12:
                    value = (
                        candidate[:4] + " " +
                        candidate[4:8] + " " +
                        candidate[8:12]
                    )

                    confidence = sum(confidences) / len(confidences)

                    return DocumentField(
                        field_name="aadhaar_number",
                        value=value,
                        confidence=confidence,
                        source_box=numeric[i][0],
                    )

                if len(candidate) > 12:
                    break

        return None

    def _extract_with_patterns(
        self,
        text: str,
        patterns: Dict[str, str],
        boxes: List[OCRBox],
    ) -> Dict[str, DocumentField]:
        """Extract fields using regex patterns."""

        fields = {}

        for field_name, pattern in patterns.items():
            match = re.search(
                pattern,
                text,
                re.IGNORECASE | re.MULTILINE,
            )

            if match:
                value = (
                    match.group(1)
                    if match.lastindex
                    else match.group(0)
                )

                value = value.strip()

                # Normalize Aadhaar number formatting.
                if field_name == "aadhaar_number":
                    digits = re.sub(r"\D", "", value)
                    if len(digits) == 12:
                        value = (
                            digits[:4] + " " +
                            digits[4:8] + " " +
                            digits[8:12]
                        )

                confidence = self._find_box_confidence(
                    value,
                    boxes,
                )

                fields[field_name] = DocumentField(
                    field_name=field_name,
                    value=value,
                    confidence=confidence,
                )

        return fields

    def _find_box_confidence(
        self,
        text: str,
        boxes: List[OCRBox],
    ) -> float:
        """Find the OCR box best matching the extracted text."""

        if not boxes:
            return 0.5

        text_lower = text.lower().strip()
        best_confidence = 0.5

        for box in boxes:
            box_text = box.text.lower().strip()

            if (
                text_lower in box_text
                or box_text in text_lower
            ):
                best_confidence = max(
                    best_confidence,
                    box.confidence,
                )

        return best_confidence

    def _extract_name_spatial(
        self,
        boxes: List[OCRBox],
    ) -> Optional[DocumentField]:
        """
        Extract name using spatial heuristics:
        top portion of document + longest alphabetic text.
        """

        if not boxes:
            return None

        sorted_boxes = sorted(
            boxes,
            key=lambda b: min(
                pt[1] for pt in b.bbox
            ),
        )

        cutoff = max(
            1,
            int(len(sorted_boxes) * 0.6),
        )

        candidates = sorted_boxes[:cutoff]

        name_candidates = []

        for box in candidates:
            text = box.text.strip()

            alpha_ratio = (
                sum(
                    c.isalpha() or c.isspace()
                    for c in text
                )
                / max(len(text), 1)
            )

            if alpha_ratio > 0.8 and len(text) > 3:
                keywords = {
                    "name",
                    "dob",
                    "date",
                    "address",
                    "male",
                    "female",
                    "government",
                }

                if text.lower() not in keywords:
                    name_candidates.append(box)

        if not name_candidates:
            return None

        best = max(
            name_candidates,
            key=lambda b: len(b.text),
        )

        return DocumentField(
            field_name="name",
            value=best.text.strip(),
            confidence=best.confidence,
            source_box=best,
        )
