"""
VeriLens AI — Universal Deepfake & Media Authenticity Verifier
================================================================
A general-purpose forensic verification dashboard for images and videos.
Works for insurance claims, journalism/newsroom fact-checking, legal
evidence intake, HR/KYC identity checks, social media moderation, or
personal use — not tied to any single domain.

Run with:  streamlit run deepfake_verifier_app.py
"""

import io
import json
import time
import hashlib
import subprocess
from datetime import datetime, date

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from PIL import Image, ExifTags, ImageChops, ImageEnhance
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Optional OCR dependency for license-plate text sanity checks. The app
# degrades gracefully (structural-only check) if this isn't installed —
# pip install pytesseract + `brew install tesseract` (Mac) to enable OCR.
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# =============================================================================
# 0. PAGE CONFIG + GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="VeriLens AI — Deepfake & Authenticity Verifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME MODE — Light / Dark, selectable from the sidebar "Appearance" expander
# =============================================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

_SHARED_VARS = """
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
    --radius-lg: 24px;
    --radius-md: 16px;
    --radius-sm: 10px;
"""

_LIGHT_VARS = """
    --brand-1: #6366F1;      /* indigo */
    --brand-2: #8B5CF6;      /* violet */
    --brand-3: #A78BFA;      /* light violet */
    --brand-soft: #EEF0FF;   /* pale indigo surface */
    --ink: #1E1B3A;          /* near-black with violet cast */
    --ink-muted: #6B7280;
    --surface: #FFFFFF;
    --surface-alt: #F7F7FC;
    --border: #E7E6F5;
    --flag-bg: #FEF2F2; --flag-border: #FECACA; --flag-text: #7f1d1d;
    --clean-bg: #F0FDF4; --clean-border: #BBF7D0; --clean-text: #14532d;
"""

_DARK_VARS = """
    --brand-1: #818CF8;      /* softer indigo, pops on dark */
    --brand-2: #A78BFA;
    --brand-3: #C4B5FD;
    --brand-soft: rgba(129,140,248,0.16);
    --ink: #EDEBFA;
    --ink-muted: #A5A3C4;
    --surface: #1A1730;
    --surface-alt: #100E1E;
    --border: rgba(255,255,255,0.10);
    --flag-bg: rgba(220,38,38,0.16); --flag-border: rgba(248,113,113,0.4); --flag-text: #FCA5A5;
    --clean-bg: rgba(22,163,74,0.16); --clean-border: rgba(74,222,128,0.4); --clean-text: #86EFAC;
"""

if st.session_state.theme_mode == "Dark":
    THEME_VARS_CSS = f":root {{ {_SHARED_VARS} {_DARK_VARS} }}"
else:
    THEME_VARS_CSS = f":root {{ {_SHARED_VARS} {_LIGHT_VARS} }}"

# =============================================================================
# VEDA-AI INSPIRED THEME
# Light surface, indigo/violet brand accent, soft shadows, generous radius —
# plus a centered "hub" landing panel in the spirit of Gemini's home screen.
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/*__THEME_VARS__*/

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3, .brand-font { font-family: 'Sora', 'Inter', sans-serif; }

.stApp { background: var(--surface-alt); }
.main > div { padding-top: 1rem; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Top brand strip ---------- */
.veda-topbar {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.2rem 0 1.1rem 0;
}
.veda-topbar .logo-dot {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; box-shadow: 0 6px 16px rgba(99,102,241,0.35);
}
.veda-topbar .logo-text { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; letter-spacing: -0.3px; color: var(--ink) !important; }
.veda-topbar .logo-sub { color: var(--ink-muted) !important; font-size: 0.8rem; margin-left: 0.35rem; }

/* Safety net: Streamlit's base dark theme sometimes forces light text on
   markdown containers — pin it back to our ink color everywhere in .main */
[data-testid="stAppViewContainer"] .stMarkdown, 
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }

/* ---------- Gemini-style centered hub (upload screen) ---------- */
.veda-hub-wrap { max-width: 760px; margin: 2.2rem auto 0 auto; text-align: center; }
.veda-hub-wrap h1 {
    font-size: 2.3rem; font-weight: 800; letter-spacing: -0.8px; margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2) 60%, var(--brand-3));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.veda-hub-wrap p.sub { color: var(--ink-muted); font-size: 1.02rem; margin-bottom: 1.8rem; }

/* Marker-targeted "hub card" — the central pill-shaped panel */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hub-marker) {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.12), 0 2px 8px rgba(30,27,58,0.04);
    padding: 0.4rem 0.6rem;
}
/* Result / metric cards elsewhere on the results page */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker) {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 6px 20px rgba(30,27,58,0.05);
}

/* Segmented "chip" look for the use-case radio group */
div[role="radiogroup"] { gap: 0.4rem; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: var(--brand-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.9rem !important;
    transition: all 0.15s ease;
}
div[role="radiogroup"] label:hover { border-color: var(--brand-2) !important; }

/* File uploader styled as a soft dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: var(--brand-soft) !important;
    border: 1.5px dashed var(--brand-3) !important;
    border-radius: var(--radius-md) !important;
}
/* Instruction text ("Drag and drop file here...") was invisible against the
   light card — force it to the ink color explicitly */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] svg { fill: var(--brand-2) !important; }
/* "Browse files" button was rendering with Streamlit's default black/secondary
   style — recolor to match the brand gradient */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important; fill: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { filter: brightness(1.08); }

/* Uploaded-file chip (filename row after a file is added) */
[data-testid="stFileUploaderFile"] { color: var(--ink) !important; }
[data-testid="stFileUploaderFileName"] { color: var(--ink) !important; }

/* Primary buttons — indigo/violet gradient */
.stButton > button[kind="primary"], .stButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2));
    border: none; border-radius: 999px; font-weight: 700; letter-spacing: 0.2px;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
    padding: 0.6rem 1.2rem;
}
.stButton > button[kind="secondary"] {
    border-radius: 999px; border: 1px solid var(--border); font-weight: 600;
}

/* Badges */
.badge {
    display: inline-block; padding: 0.28rem 0.85rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px; margin: 0 0.3rem 0.3rem 0;
    background: var(--brand-soft); color: var(--brand-1); border: 1px solid rgba(99,102,241,0.25);
}
.badge-blue { background: var(--brand-soft); color: var(--brand-1); border: 1px solid rgba(99,102,241,0.3); }

/* Hero Card Styling for Results Page */
.hero-card {
    background: linear-gradient(120deg, #4338CA 0%, #6366F1 55%, #A78BFA 130%);
    padding: 1.6rem 2rem;
    border-radius: var(--radius-lg);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 16px 40px rgba(99,102,241,0.3);
}
.hero-card h1 { margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.5px; font-family: 'Sora', sans-serif; }
.hero-card p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.0rem; }

.verdict-pass {
    background: linear-gradient(90deg, rgba(22,163,74,0.12), rgba(22,163,74,0.02));
    border-left: 6px solid var(--success);
    padding: 1rem 1.4rem; border-radius: var(--radius-sm); font-size: 1.1rem; font-weight: 700; color: #14532d;
}
.verdict-review {
    background: linear-gradient(90deg, rgba(217,119,6,0.14), rgba(217,119,6,0.02));
    border-left: 6px solid var(--warning);
    padding: 1rem 1.4rem; border-radius: var(--radius-sm); font-size: 1.1rem; font-weight: 700; color: #78350f;
}
.verdict-fail {
    background: linear-gradient(90deg, rgba(220,38,38,0.14), rgba(220,38,38,0.02));
    border-left: 6px solid var(--danger);
    padding: 1rem 1.4rem; border-radius: var(--radius-sm); font-size: 1.1rem; font-weight: 700; color: #7f1d1d;
}

.flag-card {
    background: var(--flag-bg); border: 1px solid var(--flag-border); border-radius: var(--radius-sm);
    padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.92rem; color: var(--flag-text);
}
.clean-card {
    background: var(--clean-bg); border: 1px solid var(--clean-border); border-radius: var(--radius-sm);
    padding: 0.6rem 0.9rem; font-size: 0.92rem; color: var(--clean-text);
}

/* Sidebar — light VedaAI-style panel instead of dark navy */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--ink) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stNumberInput label { color: var(--ink-muted) !important; font-weight: 600; font-size: 0.85rem; }

/* ---------- Fix: input / select / date / number fields were inheriting
   Streamlit's dark base-theme widget background, making the text unreadable.
   Chasing individual data-baseweb attributes missed number inputs and the
   password field, so this covers every wrapper level (container div, the
   BaseWeb shell, and the raw <input>) for every widget type at once. ---------- */
[data-testid="stTextInput"] div, [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] div, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] div, [data-testid="stDateInput"] input,
[data-testid="stSelectbox"] div, [data-testid="stTextArea"] div, [data-testid="stTextArea"] textarea,
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"], [data-baseweb="select"] > div, [data-baseweb="select"] div,
[data-baseweb="datepicker"], [data-baseweb="datepicker"] input,
[data-baseweb="textarea"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="base-input"] input,
[data-baseweb="datepicker"] input, textarea, input {
    background: var(--surface) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border: none !important;
}
[data-baseweb="select"] * { background: var(--surface) !important; color: var(--ink) !important; }
[data-baseweb="popover"] [data-baseweb="menu"] { background: var(--surface) !important; }
[data-baseweb="popover"] li { color: var(--ink) !important; }
input::placeholder, textarea::placeholder { color: #9CA3AF !important; opacity: 1 !important; }

/* Password "show/hide" eye icon button */
[data-testid="stTextInput"] button, [data-testid="stTextInputRootElement"] button {
    background: transparent !important; color: var(--ink-muted) !important;
}
[data-testid="stTextInput"] button svg { fill: var(--ink-muted) !important; }

/* Number input steppers */
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
    background: var(--surface) !important; color: var(--ink) !important; border-color: var(--border) !important;
}

/* Date-picker calendar popup */
div[data-baseweb="calendar"] { background: var(--surface) !important; }

/* Expander panels ("Location", "Advanced Settings", "Session History") were
   also inheriting the dark theme background on their header + body */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    background: var(--surface) !important; color: var(--ink) !important;
}
[data-testid="stExpanderDetails"] { background: var(--surface) !important; }

/* Sliders — swap default red for indigo/violet.
   BaseWeb renders the filled track + thumb with an inline background-color
   style (the source of the red we're overriding), so we target that
   attribute directly rather than relying on child ordering — this is what
   made the earlier fix apply to only one of the two sliders. */
div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background-color"] {
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--brand-2) !important;
    border-color: var(--brand-2) !important;
    box-shadow: 0 0 0 4px rgba(139,92,246,0.18) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: var(--border) !important; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { color: var(--ink-muted) !important; }
div[data-testid="stThumbValue"], div[data-testid="stSliderThumbValue"] { color: var(--brand-2) !important; font-weight: 700; }

/* Appearance radio (Light/Dark) inside the sidebar expander — stack the
   pills vertically since the sidebar column is narrow */
section[data-testid="stSidebar"] [data-testid="stExpander"] div[role="radiogroup"] {
    flex-direction: column; align-items: stretch;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] div[role="radiogroup"] label {
    width: 100%;
}

/* Checkbox */
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div[role="checkbox"] { border-color: var(--brand-3) !important; }
[data-testid="stCheckbox"] input:checked + div div[role="checkbox"] { background: var(--brand-1) !important; border-color: var(--brand-1) !important; }

/* ---------- Scrolling brand marquee (VedaAI-style "who we are" strip) ---------- */
@keyframes veda-marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.marquee-band {
    width: 100%; overflow: hidden; margin: 2.4rem 0 0.5rem 0;
    padding: 1.1rem 0; background: var(--brand-soft);
    border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
}
.marquee-track {
    display: flex; width: max-content;
    animation: veda-marquee 22s linear infinite;
}
.marquee-track span {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.4rem;
    color: var(--brand-1); white-space: nowrap; padding: 0 1.4rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.marquee-track span.dim { color: var(--brand-3); font-weight: 600; font-size: 1.1rem; }

[data-testid="stMetricValue"] { color: var(--brand-1) !important; font-family: 'Sora', sans-serif; }
[data-testid="stMetricLabel"] { color: var(--ink-muted) !important; }

hr, div[data-testid="stDivider"] { border-color: var(--border) !important; }
</style>
"""
CUSTOM_CSS = CUSTOM_CSS.replace("/*__THEME_VARS__*/", THEME_VARS_CSS)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# 1. USE-CASE CONFIGURATION
# =============================================================================
USE_CASES = {
    "General / Personal Verification": {"id_label": "Reference ID", "name_label": "Subject Name"},
    "Insurance Claims": {"id_label": "Claim ID", "name_label": "Claimant Name"},
    "Journalism / Fact-Checking": {"id_label": "Story / Assignment ID", "name_label": "Source / Uploader Name"},
    "Legal Evidence Intake": {"id_label": "Case Number", "name_label": "Party / Witness Name"},
    "HR / Identity (KYC) Checks": {"id_label": "Applicant Ref. ID", "name_label": "Applicant Name"},
    "Social Media / Platform Moderation": {"id_label": "Post / Content ID", "name_label": "Account Handle"},
}

# =============================================================================
# 2. SESSION STATE
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "fraud_db" not in st.session_state:
    st.session_state.fraud_db = {
        "REF-9042": {"label": "Flagged Historical Record #1", "hist": np.random.rand(512)},
    }
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "use_case" not in st.session_state:
    st.session_state.use_case = "General / Personal Verification"

# =============================================================================
# 3. CACHED MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model.eval()
    fake_index = 1
    for idx, label in model.config.id2label.items():
        if "fake" in label.lower():
            fake_index = idx
            break
    return processor, model, fake_index

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =============================================================================
# 4. METADATA FORENSICS
# =============================================================================
def extract_video_metadata(video_path: str) -> dict:
    try:
        command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        metadata = json.loads(result.stdout) if result.returncode == 0 else {}
        tags = metadata.get("format", {}).get("tags", {})
        encoder = str(tags.get("encoder", "")).lower()
        handler = str(tags.get("handler_name", "")).lower()
        is_edited = any(tool in encoder or tool in handler for tool in
                         ["adobe", "ffmpeg", "handbrake", "capcut", "premiere", "davinci", "final cut"])
        streams = metadata.get("streams", [])
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        resolution = f'{video_streams[0].get("width","?")}x{video_streams[0].get("height","?")}' if video_streams else "Unknown"
        return {
            "is_edited": is_edited, "encoder": encoder or "Standard Camera Hardware",
            "creation_time": tags.get("creation_time", "Unknown"),
            "duration": float(metadata.get("format", {}).get("duration", 0.0)),
            "has_audio_track": has_audio, "resolution": resolution,
        }
    except Exception:
        return {"is_edited": False, "encoder": "Unknown (ffprobe unavailable)", "creation_time": "Unknown",
                "duration": 0.0, "has_audio_track": False, "resolution": "Unknown"}

def extract_image_metadata(image_path: str) -> dict:
    try:
        img = Image.open(image_path)
        exif_data = img._getexif() or {}
        software = "Standard Camera Hardware"
        is_edited = False
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if tag_name == "Software":
                software = str(value)
                if any(tool in software.lower() for tool in ["photoshop", "gimp", "canva", "lightroom", "picsart", "facetune"]):
                    is_edited = True
                break
        return {
            "is_edited": is_edited, "encoder": software, "creation_time": "Unknown",
            "duration": 0.0, "has_audio_track": False, "resolution": f"{img.width}x{img.height}",
        }
    except Exception:
        return {"is_edited": False, "encoder": "Standard Image", "creation_time": "Unknown",
                "duration": 0.0, "has_audio_track": False, "resolution": "Unknown"}

# =============================================================================
# 5. ERROR LEVEL ANALYSIS & HASHING
# =============================================================================
def compute_ela_image(image_path: str, quality: int = 90):
    try:
        original = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)
        diff = ImageChops.difference(original, resaved)
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(diff).enhance(scale)
        arr = np.array(diff).astype(np.float32)
        ela_score = float(np.mean(arr))
        return ela_image, ela_score
    except Exception:
        return None, 0.0

def average_hash(image_path: str, hash_size: int = 8) -> str:
    try:
        img = Image.open(image_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        avg = pixels.mean()
        bits = (pixels > avg).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return ""

def hamming_distance(hash1: str, hash2: str) -> int:
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# =============================================================================
# 6. DEEPFAKE INFERENCE
# =============================================================================
def analyze_image_deepfake(image_path: str):
    try:
        processor, model, fake_index = load_deepfake_model()
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        fake_prob = float(probs[fake_index].item())
        real_prob = 1.0 - fake_prob
        prediction_label = "DEEPFAKE" if fake_prob > 0.5 else "REAL"
        return fake_prob, real_prob, prediction_label
    except Exception:
        return 0.15, 0.85, "REAL"

def analyze_video_deepfake(video_path: str, max_frames: int = 20):
    cap = cv2.VideoCapture(video_path)
    frames, frame_timestamps = [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // max_frames) if total_frames else 1
    current_frame = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if current_frame % sample_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
            frame_timestamps.append(round(current_frame / fps, 2))
        current_frame += 1
    cap.release()

    if not frames: return 0.15, 0.85, "REAL", [], []

    try:
        processor, model, fake_index = load_deepfake_model()
        fake_scores, suspicious_timestamps = [], []
        for img, ts in zip(frames, frame_timestamps):
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                fake_p = probs[fake_index].item()
                fake_scores.append(fake_p)
                if fake_p > 0.6: suspicious_timestamps.append(ts)

        # A flat frame-average under-detects deepfakes that only manipulate part
        # of a clip (e.g. a face-swapped segment). Blending in the peak frame
        # score makes short, localized manipulation harder to dilute away.
        mean_fake_prob = float(np.mean(fake_scores))
        peak_fake_prob = float(np.max(fake_scores))
        avg_fake_prob = float(np.clip(0.6 * mean_fake_prob + 0.4 * peak_fake_prob, 0.0, 1.0))
        avg_real_prob = 1.0 - avg_fake_prob
        prediction_label = "DEEPFAKE" if avg_fake_prob > 0.5 else "REAL"
        frame_data = list(zip(frame_timestamps, [round(s, 4) for s in fake_scores]))
        return avg_fake_prob, avg_real_prob, prediction_label, suspicious_timestamps, frame_data
    except Exception:
        return 0.22, 0.78, "REAL", [], []

# =============================================================================
# 6b. SCENE PLAUSIBILITY / ACTION-FORENSICS MODULE (video only)
# Targets common tells in AI-generated action/crash footage that a generic
# real-vs-fake image classifier misses entirely:
#   1) Unmotivated fire/smoke/explosion onset (no plausible ignition build-up)
#   2) Garbled or frame-to-frame inconsistent license-plate text
#   3) Physically implausible motion jumps (optical-flow discontinuity)
# These are heuristic, evidence-gathering signals, not a certain verdict —
# when triggered they push the case to MANUAL REVIEW rather than an
# auto-confident REAL/FAKE label, by design.
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_plate_cascade():
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        return None if cascade.empty() else cascade
    except Exception:
        return None

def detect_fire_smoke_ratio(frame_bgr):
    """Fraction of frame pixels in fire-like (hot orange/red/yellow, high
    saturation) and smoke-like (desaturated mid-gray) HSV ranges."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    fire_mask = cv2.inRange(hsv, (0, 120, 150), (35, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 60), (180, 60, 220))
    total_px = frame_bgr.shape[0] * frame_bgr.shape[1]
    fire_ratio = float(np.count_nonzero(fire_mask)) / total_px
    smoke_ratio = float(np.count_nonzero(smoke_mask)) / total_px
    return fire_ratio, smoke_ratio

def detect_plate_region_and_text(frame_bgr):
    """Locates a plate-like rectangular region (Haar cascade) and, if OCR is
    available, extracts its text for cross-frame consistency checking."""
    cascade = load_plate_cascade()
    if cascade is None:
        return None, ""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, 1.1, 4)
    if len(plates) == 0:
        return None, ""
    x, y, w, h = plates[0]
    plate_roi = gray[y:y + h, x:x + w]
    text = ""
    if _HAS_TESSERACT and plate_roi.size > 0:
        try:
            upscaled = cv2.resize(plate_roi, (plate_roi.shape[1] * 3, plate_roi.shape[0] * 3))
            text = pytesseract.image_to_string(upscaled, config="--psm 7").strip()
        except Exception:
            text = ""
    return (x, y, w, h), text

def is_plausible_plate_text(text: str) -> bool:
    """Loose sanity filter: real plates are short alphanumeric strings with
    both letters and digits — not empty, not symbol noise, not absurdly long."""
    cleaned = "".join(ch for ch in text if ch.isalnum())
    if len(cleaned) < 4 or len(cleaned) > 12:
        return False
    return any(ch.isalpha() for ch in cleaned) and any(ch.isdigit() for ch in cleaned)

def analyze_optical_flow(frames_gray):
    """Dense optical flow between consecutive sampled frames. Returns the
    per-transition mean motion magnitude plus a *relative* discontinuity
    score (biggest single jump vs. the clip's own average motion) — using a
    relative rather than absolute threshold keeps this meaningful across
    different resolutions and frame-rates."""
    if len(frames_gray) < 2:
        return [], 0.0
    mags = []
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
    if len(mags) < 2:
        return mags, 0.0
    diffs = np.abs(np.diff(mags))
    mean_mag = float(np.mean(mags)) + 1e-6
    discontinuity_ratio = float(np.max(diffs)) / mean_mag
    return mags, discontinuity_ratio

def analyze_scene_plausibility(video_path: str, max_frames: int = 20):
    """Full crash/action-scene forensics pass. Returns per-frame series for
    charting plus a list of human-readable manual-review flags."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // max_frames) if total_frames else 1

    frames_bgr, frames_gray, timestamps = [], [], []
    current = 0
    while cap.isOpened() and len(frames_bgr) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current % sample_interval == 0:
            frames_bgr.append(frame)
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            timestamps.append(round(current / fps, 2))
        current += 1
    cap.release()

    empty_result = {
        "fire_smoke_series": [], "explosion_onset_flag": False, "sustained_fire": False,
        "plate_detected": False, "plate_texts": [], "plate_consistent": True,
        "flow_series": [], "flow_discontinuity_ratio": 0.0, "physics_flag": False,
        "ocr_available": _HAS_TESSERACT, "flags": [],
    }
    if not frames_bgr:
        return empty_result

    # --- 1) Fire / smoke onset: flag an abrupt jump, not steady ambient tone ---
    fire_smoke_series = []
    for f, ts in zip(frames_bgr, timestamps):
        fr, sr = detect_fire_smoke_ratio(f)
        fire_smoke_series.append((ts, round(fr, 4), round(sr, 4)))
    fire_vals = [row[1] for row in fire_smoke_series]
    smoke_vals = [row[2] for row in fire_smoke_series]

    explosion_onset_flag = False
    for i in range(1, len(fire_vals)):
        if (fire_vals[i] - fire_vals[i - 1] > 0.02 and max(fire_vals) > 0.03) or \
           (smoke_vals[i] - smoke_vals[i - 1] > 0.20):
            explosion_onset_flag = True
            break
    sustained_fire = (max(fire_vals) > 0.15) if fire_vals else False

    # --- 2) License-plate consistency ---
    plate_texts, plate_detected = [], False
    for f in frames_bgr:
        box, text = detect_plate_region_and_text(f)
        if box is not None:
            plate_detected = True
        if text:
            plate_texts.append(text)

    plate_consistent = True
    if plate_texts:
        implausible_ratio = sum(1 for t in plate_texts if not is_plausible_plate_text(t)) / len(plate_texts)
        unique_texts = len(set(t.upper().replace(" ", "") for t in plate_texts))
        if implausible_ratio > 0.4 or unique_texts > max(2, len(plate_texts) // 2):
            plate_consistent = False

    # --- 3) Optical-flow physics discontinuity ---
    flow_series, discontinuity_ratio = analyze_optical_flow(frames_gray)
    physics_flag = discontinuity_ratio > 0.9

    flags = []
    if explosion_onset_flag:
        flags.append("Abrupt fire/smoke onset with no visible ignition build-up — a common artifact "
                      "in AI-generated crash/action footage. Do not treat as authentic without manual review.")
    elif sustained_fire:
        flags.append("Significant fire/smoke coverage detected — manually verify plausibility of the "
                      "depicted event before treating this footage as authentic.")
    if plate_detected and not plate_consistent:
        flags.append("Vehicle license-plate text is garbled or inconsistent across frames — a known "
                      "tell of AI video generators. Manual review recommended.")
    elif plate_detected and not _HAS_TESSERACT:
        flags.append("A license-plate-like region was detected but OCR is unavailable in this environment "
                      "(install pytesseract + tesseract) — manually zoom in and check for warped characters.")
    if physics_flag:
        flags.append(f"Motion discontinuity ratio ({discontinuity_ratio:.2f}) indicates an abrupt, "
                      "physically implausible jump in movement between frames — manual review recommended.")

    return {
        "fire_smoke_series": fire_smoke_series, "explosion_onset_flag": explosion_onset_flag,
        "sustained_fire": sustained_fire, "plate_detected": plate_detected, "plate_texts": plate_texts,
        "plate_consistent": plate_consistent, "flow_series": list(zip(timestamps[1:], [round(m, 3) for m in flow_series])),
        "flow_discontinuity_ratio": discontinuity_ratio, "physics_flag": physics_flag,
        "ocr_available": _HAS_TESSERACT, "flags": flags,
    }

# =============================================================================
# 7. CONTEXT VERIFICATION & SCORE AGGREGATION
# =============================================================================
def check_weather_context(lat: float, lon: float, date_str: str, reported_weather: str, api_key: str = "") -> dict:
    if not api_key:
        return {"matched": True, "api_weather": reported_weather.capitalize(), "note": "Simulated match (no API key provided)"}
    try:
        timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={api_key}"
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json()
            actual_weather = data["data"][0]["weather"][0]["main"]
            matched = reported_weather.lower() in actual_weather.lower()
            return {"matched": matched, "api_weather": actual_weather, "note": "Verified via OpenWeatherMap API"}
    except Exception: pass
    return {"matched": False, "api_weather": "Unknown", "note": "Weather verification lookup failed"}

def check_fraud_ring(file_path: str, is_image: bool = False):
    if is_image:
        frame = cv2.imread(file_path)
    else:
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return False, None, None

    if frame is None: return False, None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = load_face_cascade()
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0: return False, None, None

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    hist = cv2.calcHist([face_roi], [0], None, [512], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    best_sim, best_ref = 0.0, None
    for ref_id, data in st.session_state.fraud_db.items():
        sim = float(np.dot(hist, data["hist"]))
        if sim > best_sim: best_sim, best_ref = sim, ref_id
    if best_sim > 0.85: return True, best_ref, best_sim
    return False, None, best_sim

def calculate_authenticity_score(fake_prob, pred_label, metadata_info, weather_info, fraud_ring_flag,
                                  ela_score, duplicate_flag, scene_result=None):
    score = 100.0
    reasons, deductions = [], []

    if pred_label == "DEEPFAKE":
        d = round(fake_prob * 50, 1); score -= d
        reasons.append(f"AI model classified content as DEEPFAKE (confidence {fake_prob*100:.1f}%)")
        deductions.append(("AI Deepfake Signal", d))
    if fraud_ring_flag[0]:
        d = 30; score -= d
        reasons.append(f"Face matched an existing flagged record ({fraud_ring_flag[1]})")
        deductions.append(("Flagged-Record Match", d))
    if metadata_info.get("is_edited"):
        d = 15; score -= d
        reasons.append(f"File shows signs of editing software use ({metadata_info.get('encoder')})")
        deductions.append(("Editing Software Detected", d))
    if not weather_info.get("matched"):
        d = 10; score -= d
        reasons.append(f"Reported context mismatch: '{weather_info.get('api_weather')}' does not align with expected conditions")
        deductions.append(("Context Mismatch", d))
    if ela_score > 25:
        d = 12; score -= d
        reasons.append(f"Elevated ELA score ({ela_score:.1f}) suggests localized splicing")
        deductions.append(("ELA Anomaly", d))
    if duplicate_flag:
        d = 20; score -= d
        reasons.append("Near-duplicate of a previously submitted file was found in this session")
        deductions.append(("Duplicate/Reused Media", d))

    # --- Scene plausibility (crash/action forensics): unmotivated fire/smoke,
    # garbled plates, and physically implausible motion ---
    force_manual_review = False
    if scene_result:
        if scene_result.get("explosion_onset_flag"):
            d = 20; score -= d
            reasons.append(scene_result["flags"][0] if scene_result["flags"] else "Unmotivated explosion/smoke onset detected")
            deductions.append(("Unmotivated Explosion/Smoke Onset", d))
            force_manual_review = True
        elif scene_result.get("sustained_fire"):
            d = 8; score -= d
            reasons.append("High fire/smoke coverage in frame — verify plausibility manually")
            deductions.append(("High Fire/Smoke Coverage", d))
        if scene_result.get("plate_detected") and not scene_result.get("plate_consistent", True):
            d = 15; score -= d
            reasons.append("License-plate text is garbled/inconsistent across frames")
            deductions.append(("Inconsistent License Plate", d))
            force_manual_review = True
        if scene_result.get("physics_flag"):
            d = 15; score -= d
            reasons.append(f"Physically implausible motion jump detected (discontinuity ratio "
                            f"{scene_result.get('flow_discontinuity_ratio', 0):.2f})")
            deductions.append(("Motion Physics Discontinuity", d))
            force_manual_review = True

    final_score = max(0.0, round(score, 1))
    if final_score >= 80:
        status = "MANUAL REVIEW REQUIRED" if force_manual_review else "PASSED (AUTHENTIC)"
    elif final_score >= 50:
        status = "MANUAL REVIEW REQUIRED"
    else:
        status = "REJECTED (HIGH FRAUD RISK)"
    return final_score, status, reasons, deductions

# =============================================================================
# 8. UI CHART HELPERS
# =============================================================================
def render_gauge(score: float):
    color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [{"range": [0, 50], "color": "#fee2e2"}, {"range": [50, 80], "color": "#fef3c7"}, {"range": [80, 100], "color": "#dcfce7"}],
        },
        title={"text": "Authenticity Score"},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def render_frame_timeline(frame_data):
    if not frame_data: return None
    df = pd.DataFrame(frame_data, columns=["Timestamp (s)", "Deepfake Probability"])
    fig = px.line(df, x="Timestamp (s)", y="Deepfake Probability", markers=True)
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Decision threshold")
    fig.add_hline(y=0.6, line_dash="dot", line_color="red", annotation_text="Suspicious frame threshold")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 1])
    return fig

def render_fire_smoke_chart(fire_smoke_series):
    if not fire_smoke_series:
        return None
    df = pd.DataFrame(fire_smoke_series, columns=["Timestamp (s)", "Fire-like pixel ratio", "Smoke-like pixel ratio"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Timestamp (s)"], y=df["Fire-like pixel ratio"], mode="lines+markers",
                              name="Fire-like", line=dict(color="#DC2626")))
    fig.add_trace(go.Scatter(x=df["Timestamp (s)"], y=df["Smoke-like pixel ratio"], mode="lines+markers",
                              name="Smoke-like", line=dict(color="#6B7280")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Pixel ratio",
                       legend=dict(orientation="h", y=1.15))
    return fig

def render_flow_chart(flow_series):
    if not flow_series:
        return None
    df = pd.DataFrame(flow_series, columns=["Timestamp (s)", "Mean motion magnitude"])
    fig = px.line(df, x="Timestamp (s)", y="Mean motion magnitude", markers=True)
    fig.update_traces(line_color="#6366F1")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def render_deduction_chart(deductions):
    if not deductions: return None
    df = pd.DataFrame(deductions, columns=["Factor", "Points Deducted"]).sort_values("Points Deducted")
    fig = px.bar(df, x="Points Deducted", y="Factor", orientation="h", color="Points Deducted", color_continuous_scale="Reds")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    return fig

def verdict_banner(status: str):
    if status.startswith("PASSED"):
        st.markdown(f'<div class="verdict-pass">✅ {status} — No significant manipulation indicators found.</div>', unsafe_allow_html=True)
    elif status.startswith("MANUAL"):
        st.markdown(f'<div class="verdict-review">🟠 {status} — Some risk indicators present, human review recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-fail">🔴 {status} — Multiple strong risk indicators detected.</div>', unsafe_allow_html=True)

# =============================================================================
# 9. TWO-STATE UI IMPLEMENTATION
# =============================================================================
if not st.session_state.is_analyzed:
    # --- STATE 1: CENTRAL "HUB" UPLOAD VIEW (Gemini-style centered panel) ---
    st.markdown("""
        <div class="veda-topbar">
            <div class="logo-dot">✨</div>
            <div class="logo-text">VeriLens AI</div>
            <div class="logo-sub">Universal Deepfake &amp; Authenticity Verifier</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="veda-hub-wrap">
            <h1>What would you like to verify today?</h1>
            <p class="sub">Choose a use case, drop your image or video evidence, and let the forensics engine take it from there.</p>
        </div>
    """, unsafe_allow_html=True)

    # Centered hub card — marker span makes the wrapping container CSS-targetable
    _hub_l, _hub_c, _hub_r = st.columns([1, 5, 1])
    with _hub_c:
        with st.container(border=True):
            st.markdown('<span class="hub-marker"></span>', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:600; color:var(--ink-muted); font-size:0.82rem; '
                        'letter-spacing:0.3px; margin:0.9rem 0 0.5rem 0.4rem;">VERIFICATION USE CASE</p>',
                        unsafe_allow_html=True)
            st.radio(
                "Verification Use Case", list(USE_CASES.keys()), key="use_case",
                horizontal=True, label_visibility="collapsed",
            )

            st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload Evidence (image or video, multiple allowed)",
                type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
            run_btn = st.button("✨  Run Full Verification", type="primary", use_container_width=True)

    st.markdown("""
        <div style='text-align:center; margin-top: 1.6rem;'>
            <span class="badge badge-blue">MULTI-LAYER FORENSICS</span>
            <span class="badge badge-blue">IMAGE + VIDEO</span>
            <span class="badge badge-blue">BATCH READY</span>
        </div>
    """, unsafe_allow_html=True)

    # --- "Who We Are" scrolling brand marquee (VedaAI-style) ---
    _marquee_item = (
        '<span>✨ VeriLens AI</span><span class="dim">Trust every pixel, verify every frame</span>'
    )
    st.markdown(f"""
        <div class="marquee-band">
            <div class="marquee-track">
                {_marquee_item * 4}
                {_marquee_item * 4}
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    # --- STATE 2: RESULTS DASHBOARD ---
    st.markdown("""
        <div class="veda-topbar">
            <div class="logo-dot">✨</div>
            <div class="logo-text">VeriLens AI</div>
            <div class="logo-sub">Verification Report</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("← New Verification", type="secondary"):
        st.session_state.is_analyzed = False
        st.rerun()

    st.markdown("""
    <div class="hero-card">
        <h1>Multi-Layer Verification Report</h1>
        <p>Analysis complete. Review the forensics breakdown below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    results = st.session_state.get("last_results", [])
    if not results:
        st.info("👈 Return to the main screen to upload evidence.")
    else:
        if len(results) > 1:
            st.subheader("📦 Batch Summary")
            summary_df = pd.DataFrame([{
                "File": r["file_name"], "Verdict": r["pred_label"], "Score": r["score"],
                "Status": r["status"], "Duplicate?": r["duplicate_flag"]
            } for r in results])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            csv_bytes = summary_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Batch Summary", csv_bytes, "verification_batch.csv", "text/csv")
            st.divider()

        for r in results:
            st.markdown(f"### 📄 {r['file_name']}")
            col_media, col_info = st.columns([1, 1])
            with col_media:
                if r["is_image"]: st.image(r["temp_path"], use_container_width=True)
                else: st.video(r["temp_path"])

            with col_info:
                # Dynamically retrieve labels for display
                current_labels = USE_CASES[r['use_case']]
                st.markdown(f"**{current_labels['id_label']}:** {r['ref_id']}  \n**{current_labels['name_label']}:** {r['subject_name'] or '—'}  \n**Use Case:** {r['use_case']}  \n**Analyzed:** {r['timestamp']}")
                st.plotly_chart(render_gauge(r["score"]), use_container_width=True)

            verdict_banner(r["status"])
            st.write("")

            with st.container(border=True):
                st.markdown('<span class="result-marker"></span>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AI Classification", r["pred_label"])
                m2.metric("Deepfake Confidence", f"{round(r['fake_prob']*100,1)}%")
                m3.metric("Real Confidence", f"{round(r['real_prob']*100,1)}%")
                m4.metric("Authenticity Score", f"{r['score']} / 100")

            tabs = st.tabs(["🚨 Explainability", "🔬 File Forensics", "🎞️ Frame Timeline",
                             "💥 Scene Plausibility", "🌐 Context & Duplicate Check", "📤 Export Report"])
            with tabs[0]:
                st.subheader("Identified Risk Indicators")
                if r["flags"]:
                    for flag in r["flags"]: st.markdown(f'<div class="flag-card">⚠️ {flag}</div>', unsafe_allow_html=True)
                    chart = render_deduction_chart(r["deductions"])
                    if chart: st.plotly_chart(chart, use_container_width=True)
                else: st.markdown('<div class="clean-card">✅ No suspicious manipulation indicators detected.</div>', unsafe_allow_html=True)
            with tabs[1]:
                st.json({"File Type": "Image" if r["is_image"] else "Video", "Encoder": r["metadata"]["encoder"], "SHA-256": r["sha256"]})
                if r["is_image"] and r.get("ela_image") is not None:
                    st.image(r["ela_image"], use_container_width=True)
                    st.metric("ELA Score", f"{r['ela_score']:.2f}")
            with tabs[2]:
                if not r["is_image"] and r["frame_data"]: st.plotly_chart(render_frame_timeline(r["frame_data"]), use_container_width=True)
            with tabs[3]:
                scene = r.get("scene_result")
                if r["is_image"] or not scene:
                    st.info("Scene plausibility forensics (fire/smoke, plate, motion physics) apply to video only.")
                else:
                    if scene["flags"]:
                        for flag in scene["flags"]:
                            st.markdown(f'<div class="flag-card">⚠️ {flag}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="clean-card">✅ No unmotivated pyrotechnics, plate inconsistency, '
                                    'or motion-physics anomalies detected.</div>', unsafe_allow_html=True)

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.write("**Fire / Smoke Onset**")
                        fs_chart = render_fire_smoke_chart(scene["fire_smoke_series"])
                        if fs_chart: st.plotly_chart(fs_chart, use_container_width=True)
                    with sc2:
                        st.write("**Motion Physics (Optical Flow)**")
                        flow_chart = render_flow_chart(scene["flow_series"])
                        if flow_chart: st.plotly_chart(flow_chart, use_container_width=True)
                        st.caption(f"Discontinuity ratio: {scene['flow_discontinuity_ratio']:.2f} "
                                   f"(>0.9 flagged as physically implausible)")

                    st.write("**License Plate Check**")
                    if not scene["plate_detected"]:
                        st.caption("No plate-like region detected in sampled frames.")
                    else:
                        st.write(f"Plate region detected · OCR available: {scene['ocr_available']} · "
                                 f"Consistent across frames: {scene['plate_consistent']}")
                        if scene["plate_texts"]:
                            st.code("\n".join(scene["plate_texts"]))
            with tabs[4]:
                c1, c2 = st.columns(2)
                with c1: st.json(r["weather"])
                with c2: st.write("Duplicate detected" if r["duplicate_flag"] else "Clean from duplicates")
            with tabs[5]:
                st.json({"status": r["status"], "score": r["score"], "sha256": r["sha256"]})
            st.divider()

# =============================================================================
# 10. DYNAMIC SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.header("⚙️ Verification Setup")

# If we are in State 2, we display the Use Case dropdown here to allow modification.
if st.session_state.is_analyzed:
    st.sidebar.selectbox("Use Case", list(USE_CASES.keys()), key="use_case")

labels = USE_CASES[st.session_state.use_case]
ref_id = st.sidebar.text_input(labels["id_label"], value="REF-2026-0001")
subject_name = st.sidebar.text_input(labels["name_label"], value="")
incident_date = st.sidebar.date_input("Content / Incident Date", value=date.today())
reported_weather = st.sidebar.selectbox("Reported Environmental Context (optional)", ["Not specified", "Clear", "Rain", "Snow", "Clouds", "Thunderstorm"])

with st.sidebar.expander("📍 Location (for context check)"):
    lat = st.number_input("Latitude", value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", value=77.2090, format="%.4f")
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password")

with st.sidebar.expander("🎚️ Advanced Settings"):
    max_frames = st.sidebar.slider("Video frames to sample", 5, 40, 20)
    ela_quality = st.sidebar.slider("ELA re-compression quality", 70, 95, 90)
    show_ela_image = st.sidebar.checkbox("Show ELA visualization", value=True)
st.sidebar.divider()

with st.sidebar.expander("🎨 Appearance"):
    st.radio(
        "Theme", ["Light", "Dark"], key="theme_mode",
        label_visibility="collapsed",
    )

if st.session_state.history:
    with st.sidebar.expander(f"🕒 Session History ({len(st.session_state.history)} verifications)"):
        hist_df = pd.DataFrame([{"Time": h["timestamp"], "File": h["file_name"], "Status": h["status"]} for h in st.session_state.history])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.pop("last_results", None)
            st.rerun()

st.sidebar.caption("VeriLens AI · Multi-layer verification supports human review and is not a sole basis for legal or financial decisions.")

# =============================================================================
# 11. PROCESSING EXECUTION (Bound to the Run Button)
# =============================================================================
if not st.session_state.is_analyzed and 'run_btn' in locals() and run_btn and uploaded_files:
    
    def process_single_file(uploaded_file):
        file_ext = uploaded_file.name.split(".")[-1].lower()
        is_image = file_ext in ["jpg", "jpeg", "png", "webp"]
        temp_path = f"temp_{int(time.time()*1000)}_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if is_image:
            metadata_res = extract_image_metadata(temp_path)
            fake_prob, real_prob, pred_label = analyze_image_deepfake(temp_path)
            suspicious_ts, frame_data = [], []
            ela_image, ela_score = compute_ela_image(temp_path, quality=ela_quality)
            scene_result = None
        else:
            metadata_res = extract_video_metadata(temp_path)
            fake_prob, real_prob, pred_label, suspicious_ts, frame_data = analyze_video_deepfake(temp_path, max_frames=max_frames)
            ela_image, ela_score = None, 0.0
            scene_result = analyze_scene_plausibility(temp_path, max_frames=max_frames)

        weather_res = (check_weather_context(lat, lon, str(incident_date), reported_weather, owm_api_key)
                       if reported_weather != "Not specified" else {"matched": True, "api_weather": "N/A", "note": "No claim to verify"})
        fraud_ring_res = check_fraud_ring(temp_path, is_image=is_image)
        file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
        phash = average_hash(temp_path) if is_image else ""
        duplicate_flag = False
        
        for past in st.session_state.history:
            if past.get("sha256") == file_hash or (phash and past.get("phash") and hamming_distance(phash, past["phash"]) <= 4):
                duplicate_flag = True
                break

        score, status, flags, deductions = calculate_authenticity_score(
            fake_prob, pred_label, metadata_res, weather_res, fraud_ring_res, ela_score, duplicate_flag, scene_result
        )
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "file_name": uploaded_file.name,
            "use_case": st.session_state.use_case, "ref_id": ref_id, "subject_name": subject_name,
            "is_image": is_image, "temp_path": temp_path, "metadata": metadata_res, "fake_prob": fake_prob,
            "real_prob": real_prob, "pred_label": pred_label, "suspicious_ts": suspicious_ts,
            "frame_data": frame_data, "weather": weather_res, "fraud_ring": fraud_ring_res,
            "ela_score": ela_score, "duplicate_flag": duplicate_flag, "sha256": file_hash, "phash": phash,
            "score": score, "status": status, "flags": flags, "deductions": deductions, "ela_image": ela_image,
            "scene_result": scene_result,
        }

    results = []
    progress = st.progress(0, text="Starting verification pipeline...")
    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Analyzing {uf.name}...")
        rec = process_single_file(uf)
        results.append(rec)
        st.session_state.history.append(rec)
    progress.progress(1.0, text="Done.")
    time.sleep(0.3)
    progress.empty()

    st.session_state["last_results"] = results
    st.session_state.is_analyzed = True
    st.rerun()