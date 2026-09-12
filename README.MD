<div align="center">

# ◈ PramaanAI — DocuNet

**AI-powered ID document verification: tamper forensics, robust OCR, and face-match identity verification — in one pipeline.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](.)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](.)
[![PyTorch](https://img.shields.io/badge/deep%20learning-PyTorch-EE4C2C)](.)
[![InsightFace](https://img.shields.io/badge/face%20verification-InsightFace-8A2BE2)](.)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](.)

*Pramaan (प्रमाण) — Sanskrit/Hindi for "proof" or "evidence."*

</div>

---

## What is PramaanAI?

PramaanAI is a production-grade document intelligence engine built to answer one question with confidence: **is this ID real, unaltered, and does it belong to the person presenting it?**

It combines classical image forensics, a deep learning tamper classifier, multi-engine OCR, and **InsightFace-powered face verification** into a single orchestrated pipeline exposed via a CLI, a REST/WebSocket API, and an interactive dashboard.

It's built for KYC flows, onboarding checks, and any workflow where a scanned or photographed ID needs to be trusted before it's trusted.

---

## Why PramaanAI

Most "ID verification" tools stop at OCR. PramaanAI treats the image itself as evidence:

- 🔎 **Forensic-grade tamper detection** — not a single heuristic, but four independent forensic signals (ELA, noise inconsistency, JPEG ghosts, copy-move) plus a CNN classifier, cross-checked against each other.
- 🪪 **Face verification, not just face detection** — powered by **InsightFace**, matching the face on the document against a live selfie/capture to confirm the ID actually belongs to the presenter.
- 🧠 **Document-aware preprocessing** — automatic edge detection, perspective rectification, glare removal, and OCR-specific enhancement, so verification quality doesn't degrade with a bad phone photo.
- ⚡ **Real-world deployable** — FastAPI REST + WebSocket live-capture support, rate limiting, request tracing, a Streamlit dashboard, and a full CLI, out of the box.
- 🧩 **Config-driven and typed** — every threshold in the pipeline (blur, glare, ELA sensitivity, copy-move inliers, face-match threshold) is a versioned dataclass, overridable via YAML or env vars no magic numbers buried in code.

---

## Pipeline Overview

Every document that enters PramaanAI moves through a single orchestrated pipeline (`DocuNetPipeline`), with each stage timed, logged, and independently serializable:

```
                                   ┌─────────────────────┐
                                   │   Input ID Image    │
                                   └──────────┬───────────┘
                                              ▼
                          1. Quality Gate  (blur · brightness · contrast · glare · resolution)
                                              ▼
                          2. Rectification  (edge detection → homography → perspective warp)
                                              ▼
                          3. Glare Removal  (HSV highlight masking → inpainting)
                                              ▼
                          4. Enhancement    (CLAHE · denoise · Sauvola binarization for OCR)
                                              ▼
        ┌─────────────────────────────────────┴──────────────────────────────────────┐
        ▼                                                                              ▼
 5a. Forensic Tamper Suite                                                   5b. Robust OCR Engine
     • Error Level Analysis (ELA)                                               • PaddleOCR / EasyOCR
     • DCT block-artifact analysis                                               • Confidence-weighted fallback
     • Noise-inconsistency mapping                                              ▼
     • JPEG ghost detection                                              6. Field Parsing
     • Copy-move forgery detection (ORB + RANSAC)                            (name, DOB, ID no., address…)
     • CNN tamper classifier (MobileNetV3-Small + Grad-CAM)
        ▼
                                              ▼
                          7. Face Verification (InsightFace)
                             extract face embedding from ID  ⟷  match against live capture
                                              ▼
                          ┌───────────────────────────────────┐
                          │   Verdict + Full Forensic Report   │
                          └───────────────────────────────────┘
```

Every stage produces a structured, JSON-serializable result — nothing is a black box.

---

## Core Capabilities

### 🧾 Preprocessing
| Module | What it does |
|---|---|
| **Quality Gate** | Rejects or flags low-quality captures (blur, poor lighting, low resolution, excessive glare) before wasting compute downstream. |
| **Rectifier** | Canny edge detection + contour approximation + homography validation to auto-crop and perspective-correct the document. |
| **Glare Handler** | HSV-based highlight detection with inpainting to recover text/details hidden under glare. |
| **Enhancer** | CLAHE contrast correction, denoising, and Sauvola adaptive binarization tuned specifically for OCR legibility. |

### 🕵️ Forensic Tamper Detection
| Module | Technique |
|---|---|
| **ELA Detector** | Error Level Analysis + DCT block artifact scoring to surface recompression/edit artifacts. |
| **Noise Analyzer** | Block-wise noise variance mapping to catch spliced regions with inconsistent sensor noise. |
| **JPEG Ghost Detector** | Multi-quality JPEG re-compression ghosting to reveal regions saved at a different quality than the rest of the image. |
| **Copy-Move Detector** | ORB feature matching + RANSAC geometric verification to catch duplicated/cloned regions. |
| **Tamper Classifier** | MobileNetV3-Small CNN trained with Focal Loss, with Grad-CAM for visual explainability — the "does this *look* tampered" signal, cross-validated against the classical forensic suite. |

### 📄 OCR & Field Extraction
- Multi-engine OCR manager (**PaddleOCR** primary, **EasyOCR** fallback) with confidence-threshold arbitration.
- Structured **field parser** that turns raw OCR text into typed fields (name, date of birth, document number, address, etc.).

### 🪪 Face Verification *(new)*
- Built on **InsightFace**, PramaanAI extracts the facial embedding from the photo printed on the ID and compares it against a live selfie or webcam capture.
- Returns a similarity score and a pass/fail decision against a configurable match threshold — completing the loop from *"is this document real"* to *"does this document belong to this person."*
- Designed to slot into the same pipeline report as the forensic and OCR stages, so a single API call can return document authenticity **and** identity match in one response.

### 🧪 Synthetic Data Generation
- Augmentation pipeline (via `albumentations`) that programmatically injects glare, warp, blur, JPEG artifacts, noise, occlusion, and rotation into clean ID scans — used to generate labeled training data for the tamper classifier without needing a real forged-document dataset.

---

## Interfaces

PramaanAI ships with three ways to use the pipeline:

### 1. CLI
```bash
# Full verification of a single document
docunet verify --image path/to/id.jpg --output ./results

# Tamper-only analysis (skip OCR)
docunet verify --image path/to/id.jpg --skip-ocr

# Isolated ELA heatmap
docunet ela --image path/to/id.jpg --output heatmap.jpg

# Generate synthetic training data from clean scans
docunet generate --input data/clean --output data/synthetic --count 100

# Launch the API server
docunet server --host 0.0.0.0 --port 8000

# Batch benchmarking
docunet benchmark --input data/test_set --output benchmarks
```

### 2. REST + WebSocket API
Built on FastAPI, with request tracing, CORS, and rate limiting baked in.

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | Service and pipeline health check. |
| `POST /api/v1/verify` | Full pipeline run on an uploaded document image. |
| `POST /api/v1/ela-only` | Fast, forensics-only tamper check. |
| `POST /api/v1/batch` | Batch verification for multiple documents. |
| `WS /ws/live-capture` | Live-capture streaming for real-time capture guidance. |
| `/docs`, `/redoc` | Auto-generated OpenAPI documentation. |

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "file=@id_card.jpg"
```

### 3. Interactive Dashboard
A dark-mode Streamlit dashboard for visually stepping through every pipeline stage — quality metrics, rectification preview, glare/ELA/noise/ghost heatmaps, copy-move visualization, and parsed OCR fields.

```bash
streamlit run ai/dashboard.py
```

---

## Project Structure

```
ai/
├── pipeline.py              # Orchestrates the full verification pipeline
├── config.py                 # Central, typed, YAML/env-overridable configuration
├── protocols.py               # Structural typing contracts for every pipeline stage
├── cli.py                     # Click-based CLI (verify, ela, generate, server, benchmark)
├── dashboard.py                # Streamlit visual dashboard
├── exceptions.py
│
├── preprocessing/
│   ├── quality_gate.py         # Blur / brightness / glare / resolution checks
│   ├── rectifier.py             # Edge detection + homography-based perspective correction
│   ├── glare_handler.py          # Highlight detection + inpainting
│   └── enhancement.py             # CLAHE, denoising, OCR-specific binarization
│
├── forensics/
│   ├── ela_detector.py           # Error Level Analysis + DCT scoring
│   ├── ela_utils.py
│   ├── noise_analyzer.py          # Block-wise noise inconsistency mapping
│   ├── jpeg_ghost.py               # Multi-quality JPEG re-compression ghosting
│   └── copymove_detector.py         # ORB + RANSAC copy-move forgery detection
│
├── models/
│   └── tamper_classifier.py         # MobileNetV3-Small CNN + Focal Loss + Grad-CAM
│
├── ocr/
│   ├── ocr_engine.py                 # PaddleOCR / EasyOCR manager with fallback
│   └── field_parser.py                # Raw OCR → structured document fields
│
├── synthetic/
│   └── generator.py                    # Albumentations-based tamper/degradation augmentation
│
└── api/
    └── server.py                        # FastAPI REST + WebSocket service
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- (Optional) CUDA-capable GPU for faster CNN inference and face embedding extraction

### Installation
```bash
git clone <repo-url>
cd PramaanAI-Model
pip install -r requirements.txt
```

### Quick Verification
```python
from ai.config import DocuNetConfig
from ai.pipeline import DocuNetPipeline

pipeline = DocuNetPipeline(DocuNetConfig.default())
result = pipeline.process_file("path/to/id_card.jpg")

print("Tamper score:", result.ela_result.anomaly_score)
print("OCR confidence:", result.ocr_result.avg_confidence)
print(result.to_dict())
```

### Configuration
Every threshold — blur sensitivity, ELA anomaly cutoff, copy-move inlier count, face-match similarity threshold, and more — lives in a typed, immutable dataclass in `ai/config.py`, and can be overridden via YAML:

```yaml
quality_gate:
  blur_threshold: 120.0
  max_glare_ratio: 0.35

ela:
  anomaly_threshold: 0.45

ocr:
  engine: paddleocr
  confidence_threshold: 0.7
```

```python
config = DocuNetConfig.from_yaml("config.yaml")
pipeline = DocuNetPipeline(config)
```

---

## Roadmap

- [x] Multi-signal classical forensics (ELA, noise, JPEG ghost, copy-move)
- [x] CNN tamper classifier with Grad-CAM explainability
- [x] Multi-engine OCR with confidence fallback
- [x] REST + WebSocket API with live-capture support
- [x] **InsightFace-based face verification (ID photo ↔ live capture)**
- [ ] Liveness detection to harden face verification against spoofed/printed photos
- [ ] Document-type auto-classification (Aadhaar, PAN, passport, driver's license, etc.)
- [ ] Async batch processing at scale
- [ ] Model registry & versioned tamper-classifier retraining pipeline

---

## Disclaimer

PramaanAI is a decision-support tool for document and identity verification. Forensic and face-match scores should be reviewed as part of a broader verification process, not treated as an infallible, standalone verdict — especially in regulated KYC/AML contexts.

---

<div align="center">
Built with OpenCV, PyTorch, PaddleOCR/EasyOCR, InsightFace, and FastAPI.
</div>
