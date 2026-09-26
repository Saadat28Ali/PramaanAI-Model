
"""
Field parser for OCR results.

Important:
    Document type is ALWAYS supplied by the caller.

    This parser does NOT automatically classify documents.

Supported:
    - Aadhaar
    - PAN

Language focus:
    - English
    - Hindi
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from .ocr_engine import OCRBox, OCRResult


# ═════════════════════════════════════════════════════════════
# DATA MODELS
# ═════════════════════════════════════════════════════════════

@dataclass
class DocumentField:
    field_name: str
    value: str
    confidence: float
    source_box: Optional[OCRBox] = None


@dataclass
class ParsedDocument:
    fields: Dict[str, DocumentField]
    document_type: str
    overall_confidence: float
    raw_text: str
    unmatched_text: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": {
                key: {
                    "value": value.value,
                    "confidence": round(value.confidence, 4),
                }
                for key, value in self.fields.items()
            },
            "document_type": self.document_type,
            "overall_confidence": round(
                self.overall_confidence,
                4,
            ),
            "raw_text": self.raw_text,
            "unmatched_text": self.unmatched_text,
        }


# ═════════════════════════════════════════════════════════════
# COMMON PATTERNS
# ═════════════════════════════════════════════════════════════

DATE_PATTERN = r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"


# ═════════════════════════════════════════════════════════════
# DOCUMENT PATTERNS
# ═════════════════════════════════════════════════════════════

AADHAAR_PATTERNS = {
    "aadhaar_number": (
        r"(?<!\d)"
        r"(?:\d{4}[\s-]?){2}\d{4}"
        r"(?!\d)"
    ),

    "dob": (
        r"(?:"
        r"DOB|Date of Birth|Year of Birth|"
        r"जन्म तिथि|जन्मतिथि|जन्म तारीख|"
        r"जन्म वर्ष|जन्म का वर्ष"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "gender": (
        r"\b("
        r"MALE|FEMALE|Male|Female|"
        r"पुरुष|महिला|स्त्री"
        r")\b"
    ),

    "vid": (
        r"(?:VID|Virtual ID|वर्चुअल आईडी)"
        r"[:\s-]*"
        r"((?:\d{4}[\s-]?){3}\d{4})"
    ),
}


PAN_PATTERNS = {
    "pan_number": r"\b[A-Z]{5}\d{4}[A-Z]\b",

    "dob": (
        r"(?:"
        r"DOB|Date of Birth|"
        r"जन्म तिथि|जन्मतिथि|जन्म तारीख"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),
}


PASSPORT_PATTERNS = {
    "passport_number": r"\b[A-Z][0-9]{7}\b",

    "dob": (
        r"(?:"
        r"DOB|Date of Birth|"
        r"जन्म तिथि|जन्मतिथि"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "date_of_issue": (
        r"(?:"
        r"Date of Issue|Issue Date|DOI|"
        r"जारी करने की तारीख|जारी करने की तिथि|जारी तिथि"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "date_of_expiry": (
        r"(?:"
        r"Date of Expiry|Expiry Date|DOE|"
        r"समाप्ति तिथि|समाप्त होने की तिथि|"
        r"वैधता समाप्ति"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "nationality": (
        r"(?:Nationality|राष्ट्रीयता)"
        r"[:\s-]*"
        r"([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]+)"
    ),

    "sex": (
        r"(?:Sex|Gender|लिंग)"
        r"[:\s-]*"
        r"(Male|Female|M|F|पुरुष|महिला|स्त्री)"
    ),
}


DRIVERS_LICENSE_PATTERNS = {
    "license_number": (
        r"(?:"
        r"DL|D\.L\.|LIC|LICENSE|DRIVING LICENSE|"
        r"DRIVING LICENCE|"
        r"ड्राइविंग लाइसेंस|ड्राइविंग लाइसेंस नंबर|"
        r"लाइसेंस नंबर"
        r")"
        r"[:\s#-]*"
        r"([A-Z0-9][A-Z0-9/-]{5,20})"
    ),

    "dob": (
        r"(?:"
        r"DOB|Date of Birth|"
        r"जन्म तिथि|जन्मतिथि"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "date_of_issue": (
        r"(?:"
        r"Date of Issue|Issue Date|DOI|"
        r"जारी करने की तारीख|जारी तिथि"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),

    "date_of_expiry": (
        r"(?:"
        r"Date of Expiry|Expiry Date|DOE|Valid Till|Validity|"
        r"समाप्ति तिथि|वैधता|मान्य तिथि"
        r")"
        r"[:\s-]*("
        + DATE_PATTERN +
        r")"
    ),
}


# ═════════════════════════════════════════════════════════════
# FIELD PARSER
# ═════════════════════════════════════════════════════════════

class FieldParser:

    # Explicit aliases only.
    # No fuzzy matching.
    # No OCR-based classification.
    DOC_TYPE_ALIASES = {
        "aadhaar": "aadhaar",
        "aadhar": "aadhaar",
        "adhar": "aadhaar",
        "aadhaar card": "aadhaar",
        "aadhar card": "aadhaar",
        "adhar card": "aadhaar",

        "pan": "pan",
        "pan card": "pan",

        "passport": "passport",

        "dl": "drivers_license",
        "driver license": "drivers_license",
        "drivers license": "drivers_license",
        "driving license": "drivers_license",
        "driving licence": "drivers_license",
        "driver licence": "drivers_license",
        "driving license card": "drivers_license",
    }

    def __init__(self) -> None:

        self.pattern_sets = {
            "aadhaar": AADHAAR_PATTERNS,
            "pan": PAN_PATTERNS,
            "passport": PASSPORT_PATTERNS,
            "drivers_license": DRIVERS_LICENSE_PATTERNS,
        }

    # ═════════════════════════════════════════════════════════
    # DOCUMENT TYPE
    # ═════════════════════════════════════════════════════════

    def _normalize_doc_type(
        self,
        doc_type: str,
    ) -> str:

        if not isinstance(doc_type, str):
            raise ValueError(
                "doc_type must be a string."
            )

        normalized = " ".join(
            doc_type.strip().lower().split()
        )

        canonical = self.DOC_TYPE_ALIASES.get(
            normalized
        )

        if canonical is None:
            raise ValueError(
                f"Unsupported document type: '{doc_type}'. "
                f"Supported types: "
                f"{', '.join(self.pattern_sets.keys())}"
            )

        return canonical

    # ═════════════════════════════════════════════════════════
    # MAIN PARSER
    # ═════════════════════════════════════════════════════════

    def parse(
        self,
        ocr_result: OCRResult,
        doc_type: str,
    ) -> ParsedDocument:

        full_text = ocr_result.full_text or ""
        boxes = ocr_result.boxes or []

        doc_type = self._normalize_doc_type(
            doc_type
        )

        logger.info(
            f"Using user-provided document type: {doc_type}"
        )

        fields: Dict[str, DocumentField] = {}

        # -----------------------------------------------------
        # DOCUMENT-SPECIFIC REGEX
        # -----------------------------------------------------

        regex_fields = self._extract_with_patterns(
            full_text,
            self.pattern_sets[doc_type],
            boxes,
        )

        fields.update(regex_fields)

        # -----------------------------------------------------
        # DOCUMENT-SPECIFIC EXTRACTION
        # -----------------------------------------------------

        if doc_type == "aadhaar":

            self._extract_aadhaar_fields(
                fields,
                boxes,
                full_text,
            )

        elif doc_type == "pan":

            self._extract_pan_fields(
                fields,
                boxes,
            )

        elif doc_type == "passport":

            self._extract_passport_fields(
                fields,
                boxes,
            )

        elif doc_type == "drivers_license":

            self._extract_dl_fields(
                fields,
                boxes,
            )

        # -----------------------------------------------------
        # UNMATCHED OCR
        # -----------------------------------------------------

        matched_values = {
            self._normalize_text(field.value)
            for field in fields.values()
        }

        unmatched = []

        for box in boxes:

            text = box.text.strip()

            if len(text) <= 2:
                continue

            normalized = self._normalize_text(text)

            if normalized not in matched_values:
                unmatched.append(text)

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        if fields:
            overall_confidence = (
                sum(
                    field.confidence
                    for field in fields.values()
                )
                / len(fields)
            )
        else:
            overall_confidence = 0.0

        return ParsedDocument(
            fields=fields,
            document_type=doc_type,
            overall_confidence=overall_confidence,
            raw_text=full_text,
            unmatched_text=unmatched,
        )

    # ═════════════════════════════════════════════════════════
    # AADHAAR
    # ═════════════════════════════════════════════════════════

    def _extract_aadhaar_fields(
        self,
        fields: Dict[str, DocumentField],
        boxes: List[OCRBox],
        full_text: str,
    ) -> None:

        lines = self._build_lines(boxes)

        # -----------------------------------------------------
        # NAME
        # -----------------------------------------------------

        if "name" not in fields:

            name = self._extract_aadhaar_name(
                lines
            )

            if name:
                fields["name"] = name

        # -----------------------------------------------------
        # ADDRESS
        # -----------------------------------------------------

        if "address" not in fields:

            address = self._extract_aadhaar_address(
                lines
            )

            if address:
                fields["address"] = address

        # -----------------------------------------------------
        # PINCODE
        # -----------------------------------------------------

        if "pincode" not in fields:

            pincode = self._extract_pincode(
                full_text,
                boxes,
            )

            if pincode:
                fields["pincode"] = pincode

        # -----------------------------------------------------
        # AADHAAR NUMBER
        # -----------------------------------------------------

        aadhaar = self._extract_aadhaar_number(
            full_text,
            boxes,
        )

        if aadhaar:
            fields["aadhaar_number"] = aadhaar

    # ═════════════════════════════════════════════════════════
    # AADHAAR NAME
    # ═════════════════════════════════════════════════════════

    def _extract_aadhaar_name(
        self,
        lines: List[List[OCRBox]],
    ) -> Optional[DocumentField]:

        """
        Aadhaar name extraction.

        Typical OCR:

            Nikhil Kumar
            Male
            29/07/1980

        OR:

            Hanumana Ji
            DOB 01/01/1959
            Male

        We use the DOB/gender neighbourhood instead of
        simply choosing the first alphabetic OCR string.
        """

        if not lines:
            return None

        candidate_lines = []

        for index, line in enumerate(lines):

            text = self._line_text(line)

            if not text:
                continue

            has_date = bool(
                re.search(
                    DATE_PATTERN,
                    text,
                )
            )

            has_gender = self._contains_gender(
                text
            )

            if not (has_date or has_gender):
                continue

            # Current line.
            candidate_lines.append(
                (index, line)
            )

            # Previous line is often the actual name.
            if index > 0:
                candidate_lines.append(
                    (index - 1, lines[index - 1])
                )

        # -----------------------------------------------------
        # Search candidates
        # -----------------------------------------------------

        for index, line in candidate_lines:

            text = self._line_text(line)

            # Remove dates.
            cleaned = re.sub(
                DATE_PATTERN,
                " ",
                text,
                flags=re.IGNORECASE,
            )

            # Remove gender.
            cleaned = re.sub(
                r"\b(?:male|female|m|f)\b",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"(?:पुरुष|महिला|स्त्री)",
                " ",
                cleaned,
            )

            # Remove DOB labels.
            cleaned = re.sub(
                r"(?:DOB|D\.O\.B\.?|Date of Birth|"
                r"Year of Birth|जन्म तिथि|जन्मतिथि|"
                r"जन्म तारीख|जन्म वर्ष)",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = " ".join(
                cleaned.split()
            ).strip()

            # Split possible OCR fragments.
            tokens = cleaned.split()

            valid_tokens = []

            for token in tokens:

                if self._is_name_token(token):
                    valid_tokens.append(token)

            if not valid_tokens:
                continue

            candidate = " ".join(
                valid_tokens
            )

            if not self._looks_like_valid_name(
                candidate
            ):
                continue

            # Prefer names containing at least two tokens.
            if len(valid_tokens) >= 2:

                confidence = self._line_confidence(
                    line
                )

                return DocumentField(
                    field_name="name",
                    value=candidate,
                    confidence=confidence,
                    source_box=line[0],
                )

        return None

    # ═════════════════════════════════════════════════════════
    # AADHAAR ADDRESS
    # ═════════════════════════════════════════════════════════

    def _extract_aadhaar_address(
        self,
        lines: List[List[OCRBox]],
    ) -> Optional[DocumentField]:

        """
        Address extraction for layouts where OCR gives:

            29/07/1980
            123, Raj Nagar
            Delhi
            8364 5789 2230

        Address lines are collected after the DOB/gender line
        and before the Aadhaar-number line.
        """

        if not lines:
            return None

        start_index = None

        # -----------------------------------------------------
        # Find DOB/gender area.
        # -----------------------------------------------------

        for i, line in enumerate(lines):

            text = self._line_text(line)

            if re.search(
                DATE_PATTERN,
                text,
            ) or self._contains_gender(text):

                start_index = i
                break

        if start_index is None:
            return None

        address_parts = []
        address_boxes = []

        # -----------------------------------------------------
        # Collect following lines.
        # -----------------------------------------------------

        for line in lines[
            start_index + 1:
        ]:

            text = self._line_text(line)

            if not text:
                continue

            # Aadhaar number line => stop.
            digits = re.sub(
                r"\D",
                "",
                text,
            )

            if len(digits) >= 10:
                break

            # Don't accidentally consume another obvious field.
            if self._is_strong_field_label(
                text
            ):
                break

            # QR / junk / tiny fragments.
            if self._looks_like_non_address(
                text
            ):
                continue

            address_parts.append(text)
            address_boxes.extend(line)

            # Usually 1-3 lines are enough.
            if len(address_parts) >= 3:
                break

        if not address_parts:
            return None

        address = " ".join(
            address_parts
        )

        address = " ".join(
            address.split()
        )

        if len(address) < 5:
            return None

        confidence = (
            sum(
                box.confidence
                for box in address_boxes
            )
            / len(address_boxes)
            if address_boxes
            else 0.5
        )

        return DocumentField(
            field_name="address",
            value=address,
            confidence=confidence,
            source_box=(
                address_boxes[0]
                if address_boxes
                else None
            ),
        )

    # ═════════════════════════════════════════════════════════
    # AADHAAR NUMBER
    # ═════════════════════════════════════════════════════════

    def _extract_aadhaar_number(
        self,
        full_text: str,
        boxes: List[OCRBox],
    ) -> Optional[DocumentField]:

        """
        Extract Aadhaar number.

        Handles normal OCR:

            8364 5789 2230

        and the common OCR case:

            8364 5789 22308

        where an extra digit is attached to the final group.

        We only perform this because the caller already supplied
        doc_type='aadhaar'.
        """

        # -----------------------------------------------------
        # 1. Look for normal 4-4-4 grouping.
        # -----------------------------------------------------

        grouped_pattern = re.compile(
            r"(?<!\d)"
            r"(\d{4})[\s-]+"
            r"(\d{4})[\s-]+"
            r"(\d{4})"
            r"(?!\d)"
        )

        match = grouped_pattern.search(
            full_text
        )

        if match:

            value = (
                f"{match.group(1)} "
                f"{match.group(2)} "
                f"{match.group(3)}"
            )

            confidence = self._find_box_confidence(
                value,
                boxes,
            )

            return DocumentField(
                field_name="aadhaar_number",
                value=value,
                confidence=confidence,
            )

        # -----------------------------------------------------
        # 2. Handle 4-4-5 OCR grouping.
        #
        # Example:
        # 8364 5789 22308
        #
        # Treat the final group as an OCR-overread and keep
        # its first four digits.
        # -----------------------------------------------------

        overread_pattern = re.compile(
            r"(?<!\d)"
            r"(\d{4})[\s-]+"
            r"(\d{4})[\s-]+"
            r"(\d{5})"
            r"(?!\d)"
        )

        match = overread_pattern.search(
            full_text
        )

        if match:

            value = (
                f"{match.group(1)} "
                f"{match.group(2)} "
                f"{match.group(3)[:4]}"
            )

            confidence = self._find_box_confidence(
                match.group(0),
                boxes,
            )

            return DocumentField(
                field_name="aadhaar_number",
                value=value,
                confidence=confidence,
            )

        # -----------------------------------------------------
        # 3. Search OCR boxes individually.
        # -----------------------------------------------------

        for box in boxes:

            digits = re.sub(
                r"\D",
                "",
                box.text,
            )

            if len(digits) == 12:

                value = (
                    digits[:4] + " " +
                    digits[4:8] + " " +
                    digits[8:12]
                )

                return DocumentField(
                    field_name="aadhaar_number",
                    value=value,
                    confidence=box.confidence,
                    source_box=box,
                )

            if len(digits) == 13:

                value = (
                    digits[:4] + " " +
                    digits[4:8] + " " +
                    digits[8:12]
                )

                return DocumentField(
                    field_name="aadhaar_number",
                    value=value,
                    confidence=max(
                        0.0,
                        box.confidence - 0.05,
                    ),
                    source_box=box,
                )

        return None

    # ═════════════════════════════════════════════════════════
    # PINCODE
    # ═════════════════════════════════════════════════════════

    def _extract_pincode(
        self,
        full_text: str,
        boxes: List[OCRBox],
    ) -> Optional[DocumentField]:

        matches = re.findall(
            r"(?<!\d)\d{6}(?!\d)",
            full_text,
        )

        if not matches:
            return None

        value = matches[-1]

        confidence = self._find_box_confidence(
            value,
            boxes,
        )

        return DocumentField(
            field_name="pincode",
            value=value,
            confidence=confidence,
        )

    # ═════════════════════════════════════════════════════════
    # OTHER DOCUMENTS
    # ═════════════════════════════════════════════════════════

    def _extract_pan_fields(
        self,
        fields: Dict[str, DocumentField],
        boxes: List[OCRBox],
    ) -> None:
        """
        PAN-specific extraction for name and father's name.

        PAN OCR often contains noisy/distorted labels, for example:

            oot Name RAHUL MISHRA
            frat ara/ Father's Name SATENDRA MISHRA

        Therefore this method uses OCR line structure instead of
        requiring an exact label-only match.
        """

        if not boxes:
            return

        lines = self._build_lines(boxes)

        if "name" not in fields:
            name = self._extract_pan_labeled_value(
                lines,
                "name",
            )
            if name:
                fields["name"] = name

        if "father_name" not in fields:
            father_name = self._extract_pan_labeled_value(
                lines,
                "father_name",
            )
            if father_name:
                fields["father_name"] = father_name

    def _extract_pan_labeled_value(
        self,
        lines: List[List[OCRBox]],
        label_type: str,
    ) -> Optional[DocumentField]:
        """Extract a PAN person-name field from an OCR line."""

        for line in lines:
            if not line:
                continue

            line_text = self._line_text(line)
            normalized = self._normalize_text(line_text)

            if label_type == "name":
                # Never treat Father's Name as the person's name.
                if self._contains_pan_father_label(normalized):
                    continue
                if not self._contains_pan_name_label(normalized):
                    continue

            elif label_type == "father_name":
                if not self._contains_pan_father_label(normalized):
                    continue
            else:
                return None

            candidate = self._pan_value_after_label(
                line,
                label_type,
            )

            candidate = self._clean_pan_person_name(
                candidate or ""
            )

            if self._looks_like_valid_pan_person_name(candidate):
                return DocumentField(
                    field_name=label_type,
                    value=candidate,
                    confidence=self._line_confidence(line),
                    source_box=line[0],
                )

            # Some OCR engines put the label on one line and the
            # value on the following line.
            try:
                line_index = lines.index(line)
            except ValueError:
                line_index = -1

            if line_index >= 0 and line_index + 1 < len(lines):
                next_line = lines[line_index + 1]
                candidate = self._clean_pan_person_name(
                    self._line_text(next_line)
                )

                if self._looks_like_valid_pan_person_name(candidate):
                    return DocumentField(
                        field_name=label_type,
                        value=candidate,
                        confidence=self._line_confidence(next_line),
                        source_box=next_line[0],
                    )

        return None

    def _contains_pan_name_label(
        self,
        text: str,
    ) -> bool:
        """Detect a PAN 'Name' label in English or Hindi."""

        return bool(
            re.search(r"\bname\b", text, re.IGNORECASE)
        ) or "नाम" in text

    def _contains_pan_father_label(
        self,
        text: str,
    ) -> bool:
        """Detect Father's Name despite common OCR corruption."""

        normalized = self._normalize_text(text)

        if (
            "father" in normalized
            or "fathers" in normalized
            or "father's" in normalized
            or "पिता" in normalized
        ):
            return True

        # Common Tesseract fragments seen around Father's Name.
        father_ocr_variants = {
            "frat",
            "fatr",
            "fath",
            "fathr",
            "fater",
            "ather",
        }

        tokens = set(
            normalized.replace("/", " " ).split()
        )

        return bool(tokens.intersection(father_ocr_variants))

    def _pan_value_after_label(
        self,
        line: List[OCRBox],
        label_type: str,
    ) -> Optional[str]:
        """Return OCR text occurring after the PAN label."""

        if not line:
            return None

        texts = [
            box.text.strip()
            for box in line
            if box.text.strip()
        ]

        if not texts:
            return None

        if label_type == "name":
            label_end = -1

            for i, text in enumerate(texts):
                normalized = self._normalize_text(text)
                if normalized in {"name", "नाम"}:
                    label_end = i
                    break

            if label_end >= 0:
                parts = texts[label_end + 1:]
                parts = self._stop_pan_value_at_label(
                    parts,
                    "father_name",
                )
                return " ".join(parts).strip()

            return self._remove_pan_label_from_text(
                " ".join(texts),
                "name",
            )

        if label_type == "father_name":
            label_end = -1

            for i, text in enumerate(texts):
                normalized = self._normalize_text(text)

                if (
                    "father" in normalized
                    or "fathers" in normalized
                    or "father's" in normalized
                    or "पिता" in normalized
                    or normalized in {
                        "frat",
                        "fatr",
                        "fath",
                        "fathr",
                        "fater",
                        "ather",
                    }
                ):
                    label_end = i
                    continue

                # Once an actual Father's Name label has been seen,
                # a later standalone 'Name' token is also part of
                # the label rather than the value.
                if label_end >= 0 and normalized in {
                    "name",
                    "नाम",
                }:
                    label_end = i

            if label_end >= 0:
                return " ".join(
                    texts[label_end + 1:]
                ).strip()

            return self._remove_pan_label_from_text(
                " ".join(texts),
                "father_name",
            )

        return None

    def _stop_pan_value_at_label(
        self,
        parts: List[str],
        label_type: str,
    ) -> List[str]:
        result = []

        for part in parts:
            normalized = self._normalize_text(part)

            if label_type == "father_name":
                if self._contains_pan_father_label(normalized):
                    break

                if normalized in {
                    "frat",
                    "fatr",
                    "fath",
                    "fathr",
                    "fater",
                    "ather",
                }:
                    break

            result.append(part)

        return result

    def _remove_pan_label_from_text(
        self,
        text: str,
        label_type: str,
    ) -> str:
        if label_type == "name":
            text = re.sub(
                r".*?\bname\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r".*?नाम",
                "",
                text,
                count=1,
            )

        elif label_type == "father_name":
            text = re.sub(
                r".*?(?:father'?s?\s+name|father|पिता(?:\s+का\s+नाम)?)",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r".*?\b(?:frat|fatr|fath|fathr|fater|ather)\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

        return text.strip()

    def _clean_pan_person_name(
        self,
        text: str,
    ) -> str:
        text = text.strip()

        text = re.sub(
            r"^[\s:;/,\-]+|[\s:;/,\-]+$",
            "",
            text,
        )

        # Remove OCR fragments that belong to the Father's Name label.
        text = re.sub(
            r"\b(?:frat|fatr|fath|fathr|fater|ather)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\b(?:date\s+of\s+birth|dob|permanent\s+account\s+number|pan\s+card|signature)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            DATE_PATTERN,
            " ",
            text,
        )

        return " ".join(text.split()).strip()

    def _looks_like_valid_pan_person_name(
        self,
        text: str,
    ) -> bool:
        text = " ".join(text.split()).strip()

        if not text or len(text) < 3 or len(text) > 60:
            return False

        if any(char.isdigit() for char in text):
            return False

        normalized = self._normalize_text(text)

        rejected = {
            "income tax",
            "income tax department",
            "department",
            "govt",
            "govt of india",
            "government",
            "government of india",
            "india",
            "permanent account number",
            "permanent account number card",
            "pan card",
            "name",
            "father",
            "father name",
            "father's name",
            "date of birth",
            "dob",
            "signature",
            "income",
            "tax",
            "भारत",
            "भारत सरकार",
            "नाम",
            "पिता",
            "पिता का नाम",
        }

        if normalized in rejected:
            return False

        valid_tokens = []
        for token in text.split():
            token = token.strip(".,:;|()[]{}")

            if len(token) < 2:
                continue

            alpha_count = sum(
                char.isalpha()
                for char in token
            )

            if alpha_count < 2:
                continue

            valid_tokens.append(token)

        return bool(valid_tokens)

    def _extract_passport_fields(
        self,
        fields: Dict[str, DocumentField],
        boxes: List[OCRBox],
    ) -> None:

        self._extract_labeled_name(
            fields,
            boxes,
            "name",
            {
                "name",
                "full name",
                "given name",
                "नाम",
                "पूरा नाम",
            },
        )


    def _extract_dl_fields(
        self,
        fields: Dict[str, DocumentField],
        boxes: List[OCRBox],
    ) -> None:

        self._extract_labeled_name(
            fields,
            boxes,
            "name",
            {
                "name",
                "full name",
                "नाम",
                "पूरा नाम",
            },
        )

        self._extract_labeled_name(
            fields,
            boxes,
            "father_name",
            {
                "father",
                "father name",
                "father's name",
                "पिता",
                "पिता का नाम",
            },
        )

    # ═════════════════════════════════════════════════════════
    # GENERIC LABEL EXTRACTION
    # ═════════════════════════════════════════════════════════

    def _extract_labeled_name(
        self,
        fields: Dict[str, DocumentField],
        boxes: List[OCRBox],
        field_name: str,
        labels: set,
    ) -> None:

        if field_name in fields:
            return

        lines = self._build_lines(
            boxes
        )

        for i, line in enumerate(lines):

            label = self._normalize_text(
                self._line_text(line)
            )

            if label not in labels:
                continue

            if i + 1 >= len(lines):
                continue

            next_line = lines[i + 1]

            value = self._line_text(
                next_line
            )

            if not self._looks_like_valid_name(
                value
            ):
                continue

            fields[field_name] = DocumentField(
                field_name=field_name,
                value=value,
                confidence=self._line_confidence(
                    next_line
                ),
                source_box=next_line[0],
            )

            return

    # ═════════════════════════════════════════════════════════
    # REGEX EXTRACTION
    # ═════════════════════════════════════════════════════════

    def _extract_with_patterns(
        self,
        text: str,
        patterns: Dict[str, str],
        boxes: List[OCRBox],
    ) -> Dict[str, DocumentField]:

        fields = {}

        for field_name, pattern in patterns.items():

            match = re.search(
                pattern,
                text,
                re.IGNORECASE | re.MULTILINE,
            )

            if not match:
                continue

            value = (
                match.group(1)
                if match.lastindex
                else match.group(0)
            )

            value = value.strip()

            if not value:
                continue

            value = self._normalize_field_value(
                field_name,
                value,
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

    # ═════════════════════════════════════════════════════════
    # OCR LINE BUILDING
    # ═════════════════════════════════════════════════════════

    def _build_lines(
        self,
        boxes: List[OCRBox],
    ) -> List[List[OCRBox]]:

        if not boxes:
            return []

        ordered = sorted(
            boxes,
            key=lambda box: (
                self._box_y(box),
                self._box_x(box),
            ),
        )

        lines: List[List[OCRBox]] = []

        for box in ordered:

            if not box.text.strip():
                continue

            placed = False

            current_y = self._box_y(box)

            for line in lines:

                line_y = sum(
                    self._box_y(item)
                    for item in line
                ) / len(line)

                threshold = max(
                    self._box_height(box) * 0.7,
                    15,
                )

                if abs(
                    current_y - line_y
                ) <= threshold:

                    line.append(box)
                    placed = True
                    break

            if not placed:
                lines.append([box])

        for line in lines:
            line.sort(
                key=lambda box: self._box_x(box)
            )

        lines.sort(
            key=lambda line: self._box_y(line[0])
        )

        return lines

    def _line_text(
        self,
        line: List[OCRBox],
    ) -> str:

        return " ".join(
            box.text.strip()
            for box in line
            if box.text.strip()
        )

    def _line_confidence(
        self,
        line: List[OCRBox],
    ) -> float:

        if not line:
            return 0.5

        return (
            sum(
                box.confidence
                for box in line
            )
            / len(line)
        )

    # ═════════════════════════════════════════════════════════
    # NAME VALIDATION
    # ═════════════════════════════════════════════════════════

    def _is_name_token(
        self,
        token: str,
    ) -> bool:

        token = token.strip(
            ".,:;|()[]{}"
        )

        if len(token) < 2:
            return False

        # Reject obvious numeric content.
        if any(
            char.isdigit()
            for char in token
        ):
            return False

        # Unicode-aware alphabetic check.
        alpha_count = sum(
            char.isalpha()
            for char in token
        )

        if alpha_count < 2:
            return False

        return True

    def _looks_like_valid_name(
        self,
        text: str,
    ) -> bool:

        text = " ".join(
            text.split()
        ).strip()

        if not text:
            return False

        if len(text) < 3:
            return False

        if len(text) > 80:
            return False

        if self._is_rejected_name_text(
            text
        ):
            return False

        tokens = text.split()

        valid_tokens = [
            token
            for token in tokens
            if self._is_name_token(token)
        ]

        if not valid_tokens:
            return False

        return True

    def _is_rejected_name_text(
        self,
        text: str,
    ) -> bool:

        normalized = self._normalize_text(
            text
        )

        rejected = {
            "government",
            "government of india",
            "india",
            "भारत",
            "भारत सरकार",
            "government of india",
            "unique identification authority",
            "unique identification authority of india",
            "uidai",
            "aadhaar",
            "aadhar",
            "आधार",
            "male",
            "female",
            "पुरुष",
            "महिला",
            "स्त्री",
            "dob",
            "date of birth",
            "जन्म तिथि",
            "जन्मतिथि",
            "name",
            "नाम",
            "address",
            "पता",
            "my aadhaar",
            "मेरा आधार",
        }

        if normalized in rejected:
            return True

        # OCR garbage often consists of a short uppercase
        # fragment such as "HTL".
        if (
            len(text) <= 5
            and text.isascii()
            and text.isupper()
        ):
            return True

        return False

    # ═════════════════════════════════════════════════════════
    # LABEL / ADDRESS HELPERS
    # ═════════════════════════════════════════════════════════

    def _contains_gender(
        self,
        text: str,
    ) -> bool:

        return bool(
            re.search(
                r"\b(?:male|female)\b",
                text,
                re.IGNORECASE,
            )
        ) or any(
            word in text
            for word in (
                "पुरुष",
                "महिला",
                "स्त्री",
            )
        )

    def _is_strong_field_label(
        self,
        text: str,
    ) -> bool:

        normalized = self._normalize_text(
            text
        )

        return normalized in {
            "name",
            "full name",
            "नाम",
            "पूरा नाम",
            "father",
            "father name",
            "father's name",
            "पिता",
            "पिता का नाम",
            "dob",
            "date of birth",
            "जन्म तिथि",
            "जन्मतिथि",
            "gender",
            "sex",
            "लिंग",
            "male",
            "female",
            "पुरुष",
            "महिला",
            "address",
            "पता",
        }

    def _looks_like_non_address(
        self,
        text: str,
    ) -> bool:

        normalized = self._normalize_text(
            text
        )

        if normalized in {
            "government",
            "government of india",
            "भारत सरकार",
            "india",
            "आधार",
            "aadhaar",
            "aadhar",
        }:
            return True

        # Very short OCR noise.
        if len(text.strip()) <= 2:
            return True

        return False

    # ═════════════════════════════════════════════════════════
    # NORMALIZATION
    # ═════════════════════════════════════════════════════════

    def _normalize_field_value(
        self,
        field_name: str,
        value: str,
    ) -> str:

        value = " ".join(
            value.split()
        )

        if field_name == "aadhaar_number":

            digits = re.sub(
                r"\D",
                "",
                value,
            )

            if len(digits) >= 12:

                digits = digits[:12]

                value = (
                    digits[:4] + " " +
                    digits[4:8] + " " +
                    digits[8:12]
                )

        elif field_name == "pan_number":

            value = re.sub(
                r"\s+",
                "",
                value,
            ).upper()

        elif field_name in {
            "passport_number",
            "license_number",
        }:

            value = re.sub(
                r"\s+",
                "",
                value,
            ).upper()

        return value.strip()

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        return " ".join(
            text.strip().lower().split()
        )

    # ═════════════════════════════════════════════════════════
    # BOUNDING BOX HELPERS
    # ═════════════════════════════════════════════════════════

    def _box_x(
        self,
        box: OCRBox,
    ) -> float:

        return min(
            point[0]
            for point in box.bbox
        )

    def _box_y(
        self,
        box: OCRBox,
    ) -> float:

        return min(
            point[1]
            for point in box.bbox
        )

    def _box_height(
        self,
        box: OCRBox,
    ) -> float:

        ys = [
            point[1]
            for point in box.bbox
        ]

        if not ys:
            return 20.0

        return max(ys) - min(ys) or 20.0

    # ═════════════════════════════════════════════════════════
    # OCR CONFIDENCE
    # ═════════════════════════════════════════════════════════

    def _find_box_confidence(
        self,
        text: str,
        boxes: List[OCRBox],
    ) -> float:

        if not boxes:
            return 0.5

        target = self._normalize_text(
            text
        )

        best = 0.5

        for box in boxes:

            box_text = self._normalize_text(
                box.text
            )

            if not box_text:
                continue

            if (
                target in box_text
                or box_text in target
            ):
                best = max(
                    best,
                    box.confidence,
                )

        return best

