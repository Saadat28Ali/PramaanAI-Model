"""
DocuNet Dashboard — Interactive document verification UI.

Run:  streamlit run src/dashboard.py
"""

import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import DocuNetConfig
from src.pipeline import DocuNetPipeline

st.set_page_config(
    page_title="DocuNet",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◈</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BG        = "#0A0A0A"
_SURFACE   = "#111111"
_SURFACE2  = "#1A1A1A"
_BORDER    = "rgba(255,255,255,0.07)"
_TEXT      = "#EDEDED"
_MUTED     = "#666"
_ACCENT    = "#EDEDED"   # white-ish — color is used ONLY for status
_OK        = "#4ADE80"   # green-400
_WARN_CLR  = "#FBB040"   # amber-ish
_ERR       = "#F87171"   # red-400

st.markdown(f"""
<style>
/* Strip Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.viewerBadge_container__1QSob {{ display:none !important; }}

/* Body */
html, body, [class*="css"],
.stApp, .stAppViewContainer,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {{
    background: {_BG} !important;
    color: {_TEXT};
    font-family: -apple-system, 'SF Pro Text', BlinkMacSystemFont,
                 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {_SURFACE} !important;
    border-right: 1px solid {_BORDER};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

/* Main padding */
.block-container {{ padding: 2.5rem 2.5rem 4rem !important; max-width: 1400px; }}

/* Headings */
h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.02em; }}

/* Section label */
.dn-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {_MUTED};
    margin: 2rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {_BORDER};
}}

/* KPI tile */
.dn-kpi {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    height: 100%;
}}
.dn-kpi-label {{
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {_MUTED};
    margin-bottom: 0.5rem;
}}
.dn-kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {_TEXT};
    font-variant-numeric: tabular-nums;
    line-height: 1;
}}
.dn-kpi-sub {{
    font-size: 0.72rem;
    color: {_MUTED};
    margin-top: 0.35rem;
}}

/* Verdict bar */
.dn-verdict {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border: 1px solid;
    margin-bottom: 2rem;
    font-size: 0.9rem;
    font-weight: 500;
}}
.dn-verdict-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.dn-verdict-ok   {{ background: rgba(74,222,128,0.08); border-color: rgba(74,222,128,0.25); color: #a3f0bf; }}
.dn-verdict-warn {{ background: rgba(251,176,64,0.08);  border-color: rgba(251,176,64,0.25); color: #fdd083; }}
.dn-verdict-err  {{ background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.25); color: #fca5a5; }}
.dot-ok   {{ background: {_OK}; }}
.dot-warn {{ background: {_WARN_CLR}; }}
.dot-err  {{ background: {_ERR}; }}

/* Upload zone */
[data-testid="stFileUploader"] section {{
    background: {_SURFACE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 10px !important;
    padding: 3rem 2rem !important;
    transition: border-color 0.15s;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: rgba(255,255,255,0.18) !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] * {{
    color: {_MUTED} !important;
}}

/* Image */
[data-testid="stImage"] img {{
    border-radius: 8px;
    display: block;
}}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0;
    border-bottom: 1px solid {_BORDER};
    background: transparent;
}}
[data-testid="stTabs"] [role="tab"] {{
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: {_MUTED} !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1rem !important;
    background: transparent !important;
    border-radius: 0 !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {_TEXT} !important;
    border-bottom-color: {_TEXT} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ display: none; }}

/* Sidebar controls */
[data-testid="stSidebar"] label {{
    font-size: 0.82rem !important;
    color: {_MUTED} !important;
    font-weight: 500 !important;
}}
[data-testid="stToggle"] [data-testid="stWidgetLabel"] p {{
    font-size: 0.82rem;
    color: {_MUTED};
}}

/* Field list */
.dn-fields {{ width: 100%; }}
.dn-field-row {{
    display: grid;
    grid-template-columns: 160px 1fr;
    align-items: baseline;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid {_BORDER};
}}
.dn-field-row:last-child {{ border-bottom: none; }}
.dn-field-key {{
    font-size: 0.75rem;
    font-weight: 500;
    color: {_MUTED};
    letter-spacing: 0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.dn-field-val {{
    font-size: 0.88rem;
    font-weight: 500;
    color: {_TEXT};
    word-break: break-word;
}}

/* Timing bars */
.dn-timing {{ margin-bottom: 0.6rem; }}
.dn-timing-meta {{
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: {_MUTED};
    margin-bottom: 0.25rem;
}}
.dn-timing-track {{
    height: 3px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
}}
.dn-timing-fill {{
    height: 100%;
    border-radius: 2px;
    background: {_TEXT};
    opacity: 0.5;
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {_SURFACE} !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {_MUTED} !important;
}}

/* Download button */
[data-testid="stDownloadButton"] > button {{
    background: transparent !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 7px !important;
    color: {_MUTED} !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important;
    letter-spacing: 0.01em;
    transition: border-color 0.15s, color 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    border-color: rgba(255,255,255,0.25) !important;
    color: {_TEXT} !important;
    background: rgba(255,255,255,0.04) !important;
}}

/* Spinner */
[data-testid="stSpinner"] p {{ color: {_MUTED} !important; font-size: 0.82rem !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.12); border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return DocuNetPipeline(DocuNetConfig.default())


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="dn-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="dn-kpi">'
        f'<div class="dn-kpi-label">{label}</div>'
        f'<div class="dn-kpi-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _verdict(label: str, score: float, cls: str, dot_cls: str) -> str:
    return (
        f'<div class="dn-verdict {cls}">'
        f'<span class="dn-verdict-dot {dot_cls}"></span>'
        f'<strong>{label}</strong>&ensp;—&ensp;anomaly score&nbsp;<code style="background:transparent;font-size:0.85em">{score:.4f}</code>'
        f'</div>'
    )


def _timing_bars(timings: dict) -> str:
    if not timings:
        return ""
    max_ms = max(timings.values(), default=1)
    rows = []
    for stage, ms in timings.items():
        pct = (ms / max_ms) * 100 if max_ms else 0
        rows.append(
            f'<div class="dn-timing">'
            f'<div class="dn-timing-meta"><span>{stage.replace("_"," ").title()}</span><span>{ms:.0f} ms</span></div>'
            f'<div class="dn-timing-track"><div class="dn-timing-fill" style="width:{pct:.1f}%"></div></div>'
            f'</div>'
        )
    return "".join(rows)


def _field_table(fields: dict) -> str:
    rows = []
    for k, v in fields.items():
        rows.append(
            f'<div class="dn-field-row">'
            f'<span class="dn-field-key">{k.replace("_"," ").title()}</span>'
            f'<span class="dn-field-val">{v.value}</span>'
            f'</div>'
        )
    return f'<div class="dn-fields">{" ".join(rows)}</div>'


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:2rem">'
        f'DocuNet · v1.0</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:0.75rem">Options</p>',
        unsafe_allow_html=True,
    )
    skip_quality = st.toggle("Skip quality gate", value=False)
    skip_ocr     = st.toggle("Skip OCR",          value=False)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:0.75rem">Stack</p>',
        unsafe_allow_html=True,
    )
    for item in ["OpenCV", "PaddleOCR", "PyTorch", "FastAPI"]:
        st.markdown(
            f'<p style="font-size:0.78rem;color:{_MUTED};margin:0.2rem 0">{item}</p>',
            unsafe_allow_html=True,
        )



st.markdown(
    f'<h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;'
    f'color:{_TEXT};margin:0 0 0.3rem">DocuNet</h1>'
    f'<p style="font-size:0.88rem;color:{_MUTED};margin:0 0 2rem">'
    f'Document forensics — tampering detection &amp; OCR analysis</p>',
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "drop_zone",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)


if not uploaded_file:
    st.markdown(
        f'<p style="font-size:0.78rem;color:{_MUTED};text-align:center;'
        f'margin-top:1.5rem">'
        f'Upload a document image to begin analysis.</p>',
        unsafe_allow_html=True,
    )
    st.stop()


raw = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

if image is None:
    st.error("Could not decode the image file.")
    st.stop()


with st.spinner("Running analysis…"):
    pipeline = load_pipeline()
    result   = pipeline.process(
        image,
        skip_quality_gate=skip_quality,
        skip_ocr=skip_ocr,
    )


tamper_score = 0.0
is_tampered  = False
if result.ela_result:
    tamper_score = result.ela_result.anomaly_score
    is_tampered  = result.ela_result.is_tampered

ocr_conf = result.ocr_result.avg_confidence if result.ocr_result else 0.0


if is_tampered:
    st.markdown(_verdict("Tampering Detected", tamper_score, "dn-verdict-err",  "dot-err"),  unsafe_allow_html=True)
elif tamper_score > 0.25:
    st.markdown(_verdict("Suspicious",         tamper_score, "dn-verdict-warn", "dot-warn"), unsafe_allow_html=True)
elif result.ela_result:
    st.markdown(_verdict("Appears Authentic",  tamper_score, "dn-verdict-ok",   "dot-ok"),   unsafe_allow_html=True)


k1, k2, k3, k4 = st.columns(4)

with k1:
    verdict_str = "Tampered" if is_tampered else ("Suspicious" if tamper_score > 0.25 else "Authentic")
    st.markdown(_kpi("Verdict", verdict_str), unsafe_allow_html=True)

with k2:
    st.markdown(_kpi("Anomaly Score", f"{tamper_score:.4f}", "lower = more authentic"), unsafe_allow_html=True)

with k3:
    blur = f"{result.quality_report.blur_score:.0f}" if result.quality_report else "—"
    st.markdown(_kpi("Sharpness", blur, "blur score"), unsafe_allow_html=True)

with k4:
    st.markdown(_kpi("Total Time", f"{result.total_time_ms:.0f} ms"), unsafe_allow_html=True)


st.markdown('<div class="dn-label">Image Analysis</div>', unsafe_allow_html=True)

disp         = result.images.get("rectified", result.images.get("enhanced", image))
ela_overlay  = result.images.get("ela_overlay")
_hm_from_images = result.images.get("ela_heatmap")
ela_heatmap  = _hm_from_images if _hm_from_images is not None else (result.ela_result.heatmap if result.ela_result else None)
has_ela      = ela_overlay is not None or ela_heatmap is not None

if has_ela:
    ci_orig, ci_proc, ci_ela = st.columns(3, gap="medium")
else:
    ci_orig, ci_proc = st.columns(2, gap="medium")

with ci_orig:
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:0.5rem">Original</p>',
        unsafe_allow_html=True,
    )
    st.image(_bgr_to_rgb(image), use_container_width=True)

with ci_proc:
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:0.5rem">Processed</p>',
        unsafe_allow_html=True,
    )
    st.image(_bgr_to_rgb(disp), use_container_width=True)

if has_ela:
    with ci_ela:
        st.markdown(
            f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{_MUTED};margin-bottom:0.5rem">ELA</p>',
            unsafe_allow_html=True,
        )
        if ela_overlay is not None and ela_heatmap is not None:
            tab_overlay, tab_heatmap = st.tabs(["Overlay", "Heatmap"])
            with tab_overlay:
                st.image(_bgr_to_rgb(ela_overlay), use_container_width=True)
                st.caption("Heatmap blended onto original — manipulation appears as warm-coloured regions.")
            with tab_heatmap:
                st.image(_bgr_to_rgb(ela_heatmap), use_container_width=True)
                st.caption("Raw ELA map — brightness = compression anomaly intensity.")
        elif ela_overlay is not None:
            st.image(_bgr_to_rgb(ela_overlay), use_container_width=True)
            st.caption("Heatmap blended onto original.")
        else:
            st.image(_bgr_to_rgb(ela_heatmap), use_container_width=True)
            st.caption("Raw ELA map — overlay not available.")


st.markdown('<div class="dn-label">Extracted Fields</div>', unsafe_allow_html=True)

if result.parsed_document and result.parsed_document.fields:
    st.markdown(_field_table(result.parsed_document.fields), unsafe_allow_html=True)
    if result.parsed_document.raw_text:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Raw OCR text"):
            st.code(result.parsed_document.raw_text, language=None)
else:
    st.markdown(
        f'<p style="font-size:0.82rem;color:{_MUTED}">No fields extracted.</p>',
        unsafe_allow_html=True,
    )

if result.ela_result and result.ela_result.is_tampered and result.ela_result.suspicious_regions:
    st.markdown("<br>", unsafe_allow_html=True)
    regions = result.ela_result.suspicious_regions
    with st.expander(f"{len(regions)} suspicious region{'s' if len(regions) != 1 else ''}"):
        for i, r in enumerate(regions, 1):
            st.caption(
                f"Region {i} · ({r['x']}, {r['y']}) · {r['w']}×{r['h']} · intensity {r['intensity']:.1f}"
            )

if result.quality_report and not result.quality_report.passed:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Quality warnings"):
        for issue in result.quality_report.issues:
            st.caption(issue)


if result.timings:
    st.markdown('<div class="dn-label">Pipeline Latency</div>', unsafe_allow_html=True)
    st.markdown(_timing_bars(result.timings), unsafe_allow_html=True)


st.markdown('<div class="dn-label">Export</div>', unsafe_allow_html=True)

st.download_button(
    label="Download JSON report",
    data=json.dumps(result.to_dict(), indent=2, default=str),
    file_name="docunet_report.json",
    mime="application/json",
)


st.markdown(
    f'<p style="font-size:0.7rem;color:{_MUTED};margin-top:4rem">'
    f'DocuNet · All processing is local</p>',
    unsafe_allow_html=True,
)
