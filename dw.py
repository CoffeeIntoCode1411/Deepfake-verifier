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

# PDF report generation. Static chart images are rendered with matplotlib
# (no browser/kaleido dependency needed) and laid out into a branded PDF
# with reportlab. The app degrades gracefully — the "Download PDF Report"
# button simply won't appear — if these aren't installed.
# pip install reportlab matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, HRFlowable,
    )
    _HAS_PDF_LIBS = True
except Exception:
    _HAS_PDF_LIBS = False

# =============================================================================
# 0. PAGE CONFIG + GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="VeriLens AI — Deepfake & Authenticity Verifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,800&display=swap');

/*__THEME_VARS__*/

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif; color: var(--ink);
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}
h1, h2, h3, .brand-font {
    font-family: 'Fraunces', 'Sora', serif;
    font-optical-sizing: auto;
    font-feature-settings: "ss01" on, "ss02" on;
    letter-spacing: -0.01em;
}

.stApp { background: var(--surface-alt); }
.main > div { padding-top: 1rem; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Top brand strip ---------- */
.veda-topbar {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 0.3rem 0 1.2rem 0; text-align: center;
}
.veda-topbar-row { display: flex; align-items: center; justify-content: center; gap: 1rem; }
.veda-topbar .logo-dot {
    width: 46px; height: 46px; border-radius: 14px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 6px 16px rgba(99,102,241,0.35), inset 0 1px 1px rgba(255,255,255,0.55),
                inset 0 -3px 5px rgba(30,20,70,0.25), 0 0 0 6px rgba(99,102,241,0.09);
}
.veda-topbar .logo-text {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.5rem; letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2) 60%, var(--brand-3));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    color: var(--ink) !important;
    filter: drop-shadow(0 2px 10px rgba(99,102,241,0.18));
}
.veda-topbar .logo-sub {
    color: var(--ink-muted) !important; font-size: 1.02rem; font-weight: 700;
    margin-top: 0.4rem; letter-spacing: 0.1px;
}

/* Safety net: Streamlit's base dark theme sometimes forces light text on
   markdown containers — pin it back to our ink color everywhere in .main */
[data-testid="stAppViewContainer"] .stMarkdown, 
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }

/* ---------- Minimalist Command Center hero (upload screen) ---------- */
.veda-hub-wrap { max-width: 780px; margin: 3.1rem auto 0 auto; text-align: center; }
.veda-eyebrow {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 2.2px; text-transform: uppercase; color: var(--brand-1);
    background: var(--brand-soft); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 999px; padding: 0.38rem 0.9rem; margin-bottom: 1.2rem;
    box-shadow: 0 2px 6px rgba(99,102,241,0.10), inset 0 1px 0 rgba(255,255,255,0.6);
}
.veda-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); box-shadow: 0 0 0 3px rgba(22,163,74,0.18); }
.veda-hub-wrap h1 {
    font-family: 'Fraunces', serif; font-optical-sizing: auto; font-style: normal;
    font-size: 3.15rem; font-weight: 600; letter-spacing: -0.5px; line-height: 1.1; margin: 0 0 0.7rem 0;
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2) 60%, var(--brand-3));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 6px 18px rgba(99,102,241,0.22));
}
.veda-hub-wrap p.sub { color: var(--ink-muted); font-size: 1.05rem; font-weight: 450; margin: 0 auto 2.1rem auto; max-width: 560px; line-height: 1.6; }

/* Marker-targeted "hub card" — the central pill-shaped panel */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hub-marker) {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 20px 50px rgba(99,102,241,0.14), 0 2px 8px rgba(30,27,58,0.05);
    padding: 0.4rem 0.6rem;
    transition: transform 0.35s cubic-bezier(0.22,1,0.36,1), box-shadow 0.35s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hub-marker):hover {
    transform: translateY(-3px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 28px 64px rgba(99,102,241,0.18), 0 4px 12px rgba(30,27,58,0.07);
}
/* Result / metric cards elsewhere on the results page */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker) {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 6px 20px rgba(30,27,58,0.05);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker):hover {
    transform: translateY(-2px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 12px 30px rgba(30,27,58,0.08);
}
/* Each of the 4 forensic metric tiles (AI Classification / Deepfake Confidence /
   Real Confidence / Authenticity Score) is its OWN real bordered container,
   given a stable key= so Streamlit classes it directly as st-key-metric_card_*
   — this is the officially supported targeting mechanism, more reliable than
   the marker+:has() trick which wasn't reaching the element in practice.
   A pronounced 3D "bulge toward the viewer" lift shows on hover. */
div[class*="st-key-metric_card_"] {
    background: var(--surface-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.3rem 0.4rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 2px 6px rgba(30,27,58,0.04);
    transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    cursor: default;
}
div[class*="st-key-metric_card_"] [data-testid="stMetric"],
div[class*="st-key-metric_card_"] [data-testid="stMetricLabel"],
div[class*="st-key-metric_card_"] [data-testid="stMetricValue"] {
    text-align: center; justify-content: center; width: 100%;
}
div[class*="st-key-metric_card_"]:hover {
    transform: translateY(-8px) scale(1.035);
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 20px 36px rgba(99,102,241,0.24), 0 4px 10px rgba(30,27,58,0.08);
    border-color: var(--brand-2);
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

/* ---------- "Ask-bar" style Dropzone — a slim, fully-rounded capsule
   (Gemini / chat-composer style) instead of a big dashed box. The whole
   pill is one continuous click/drop target; a small paperclip glyph and
   muted placeholder copy sit inline on the left, and the round gradient
   send button (rendered right next to it in an adjoining Streamlit column)
   completes the "ask" affordance. ---------- */
.dropzone-shell { margin: 0.2rem 0 0.9rem 0; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 999px !important;
    min-height: 0 !important;
    height: 58px !important;
    display: flex !important; flex-direction: row !important;
    align-items: center !important; justify-content: flex-start !important;
    padding: 0 0.5rem 0 1.3rem !important;
    animation: none !important;
    box-shadow: 0 2px 10px rgba(30,27,58,0.06);
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within {
    border-color: var(--brand-2) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.16) !important;
}
/* Icon + instruction text laid out inline (icon, then a single muted line) —
   the built-in "Drag and drop file here" copy is swapped for a Gemini-style
   prompt line via a pseudo-element so wording isn't tied to Streamlit's
   default string, and the verbose "Limit 200MB..." sub-line moves out of
   the bar into a small caption underneath it instead. */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important; flex-direction: row !important;
    align-items: center !important; gap: 0.65rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: var(--ink-muted) !important;
    width: 20px !important; height: 20px !important;
    animation: none !important; flex-shrink: 0;
}
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child {
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 0 !important; line-height: 1.3;
}
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child::after {
    content: "Attach or drag & drop an image or video here to verify";
    font-size: 0.95rem; color: var(--ink-muted); font-weight: 500;
}
/* The native "Browse files" button is redundant once the whole pill is a
   click target — drop it so the bar reads as a clean single-line input
   rather than icon + text + button crowding one capsule. */
[data-testid="stFileUploaderDropzone"] button { display: none !important; }

/* ---------- Circular "send" button that completes the ask-bar, sitting in
   the narrow column immediately to the right of the pill above. ---------- */
div[class*="st-key-gemini_send_btn"] .stButton { display: flex; justify-content: center; }
div[class*="st-key-gemini_send_btn"] .stButton > button,
div[class*="st-key-gemini_send_btn"] .stButton > button[kind="primary"] {
    width: 52px !important; height: 52px !important; min-width: 52px !important;
    border-radius: 50% !important; padding: 0 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 1.15rem !important; line-height: 1 !important;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2)) !important;
    box-shadow: 0 8px 18px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.3) !important;
    margin: 0 auto;
}
div[class*="st-key-gemini_send_btn"] .stButton > button:hover { transform: translateY(-1px) scale(1.03); }
.gemini-bar-hint {
    text-align: center; color: var(--ink-muted); font-size: 0.78rem;
    margin: 0.6rem 0 0 0; font-weight: 500;
}

/* Uploaded-file chip → styled as a proper preview card, not a bare filename
   row: bordered surface, generous padding, a settled (non-animated) state
   that visually confirms "this file is locked in and ready" */
[data-testid="stFileUploaderFile"] {
    color: var(--ink) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1rem !important;
    margin-top: 0.7rem !important;
    box-shadow: 0 6px 18px rgba(30,27,58,0.06);
}
[data-testid="stFileUploaderFileName"] { color: var(--ink) !important; font-weight: 600 !important; }
[data-testid="stFileUploaderFile"] small { color: var(--ink-muted) !important; }
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderDeleteBtn"] button svg { fill: var(--ink-muted) !important; }

/* Primary buttons — indigo/violet gradient, tactile 3D "keycap" press */
.stButton > button[kind="primary"], .stButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(180deg, var(--brand-1), var(--brand-2));
    border: none; border-radius: 999px; font-weight: 700; letter-spacing: 0.2px;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -3px 6px rgba(30,20,70,0.18);
    padding: 0.6rem 1.2rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
}
.stButton > button[kind="primary"]:hover, .stButton > button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px);
    filter: brightness(1.04);
}
.stButton > button[kind="primary"]:active, .stButton > button[kind="primaryFormSubmit"]:active {
    transform: translateY(2px);
    box-shadow: 0 3px 10px rgba(99,102,241,0.3), inset 0 1px 0 rgba(255,255,255,0.15), inset 0 -1px 2px rgba(30,20,70,0.12);
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border-radius: 999px; border: 1px solid var(--border); font-weight: 600;
    box-shadow: 0 2px 6px rgba(30,27,58,0.04);
    transition: transform 0.12s ease, border-color 0.15s ease, color 0.15s ease;
}
.stButton > button[kind="secondary"]:hover { border-color: var(--brand-2) !important; color: var(--brand-1) !important; transform: translateY(-1px); }
.stButton > button[kind="secondary"]:active { transform: translateY(1px); }

/* Filename caption directly above its photo/video — bold, no gap before the media */
.media-caption {
    font-weight: 700; font-size: 1.05rem; color: var(--ink);
    margin: 0 0 -0.6rem 0; padding: 0;
}

/* "Details Card" for per-file metadata (Reference ID, Subject Name, etc.) —
   a crisp two-column grid instead of loose floating markdown text */
.details-card {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.15rem 1.35rem;
    display: grid; grid-template-columns: auto 1fr;
    column-gap: 1.3rem; row-gap: 0.7rem;
    align-items: baseline;
    box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 4px 14px rgba(30,27,58,0.05);
}
.dc-row { display: contents; }
.dc-label { color: var(--ink-muted); font-weight: 600; font-size: 0.87rem; white-space: nowrap; }
.dc-value { color: var(--ink); font-weight: 700; font-size: 0.96rem; }

/* Badges */
.badge {
    display: inline-block; padding: 0.28rem 0.85rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px; margin: 0 0.3rem 0.3rem 0;
    background: var(--brand-soft); color: var(--brand-1); border: 1px solid rgba(99,102,241,0.25);
    box-shadow: 0 2px 6px rgba(99,102,241,0.10), inset 0 1px 0 rgba(255,255,255,0.55);
}
.badge-blue { background: var(--brand-soft); color: var(--brand-1); border: 1px solid rgba(99,102,241,0.3); }

/* Hero Card Styling for Results Page */
.hero-card {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #3730A3 0%, #4338CA 30%, #6366F1 60%, #8B5CF6 90%, #A78BFA 130%);
    padding: 1.9rem 2.2rem;
    border-radius: var(--radius-lg);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 48px rgba(99,102,241,0.32), inset 0 1px 0 rgba(255,255,255,0.25);
}
/* Subtle scan-grid texture, echoing the 3D holographic floor elsewhere */
.hero-card::before {
    content: ""; position: absolute; inset: 0; z-index: 0; pointer-events: none;
    background-image: linear-gradient(rgba(255,255,255,0.09) 1px, transparent 1px),
                       linear-gradient(90deg, rgba(255,255,255,0.09) 1px, transparent 1px);
    background-size: 36px 36px;
    -webkit-mask-image: linear-gradient(120deg, black, transparent 75%);
    mask-image: linear-gradient(120deg, black, transparent 75%);
}
/* Two soft glow blobs for depth */
.hero-card::after {
    content: ""; position: absolute; z-index: 0; pointer-events: none;
    top: -55%; right: -8%; width: 340px; height: 340px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 70%);
    filter: blur(6px);
}
.hero-card .glow-blob-2 {
    content: ""; position: absolute; z-index: 0; pointer-events: none;
    bottom: -60%; left: -6%; width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(167,139,250,0.45), transparent 70%);
    filter: blur(10px);
}
.hero-card h1, .hero-card p { position: relative; z-index: 1; }
.hero-card h1 { margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.3px; font-family: 'Fraunces', serif; text-shadow: 0 2px 12px rgba(0,0,0,0.18); }
.hero-card p { margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1.03rem; }

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

/* ---------- Sidebar fully removed — everything it held now lives in-page
   (Verification Details expander on the hub, Appearance/History as icon
   popovers) — hide the sidebar panel AND its little collapse-arrow toggle
   so no trace of it remains ---------- */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Pin the "⋮" trigger's column to the true browser viewport corner via
   position:fixed — Streamlit's wide layout centers its content with a
   max-width, so the column's own right edge was still visibly inset from
   the real window edge. Streamlit assigns each column an explicit width
   percentage (not flex-grow auto-distribution), so pulling this one out of
   normal flow does not affect the other columns' widths — the 1:10:1 topbar
   ratio still centers the logo correctly. */
div[data-testid="column"]:has(.topbar-icon-marker) {
    position: fixed !important; top: 90px; right: 24px; z-index: 999;
    width: auto !important; min-width: 0 !important; flex: none !important;
    display: flex; justify-content: flex-end;
}

/* Single "⋮" utility menu (History + Appearance combined) */
[data-testid="stPopover"] button {
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    font-weight: 900 !important;
    font-size: 1.15rem !important;
    letter-spacing: 1px;
    padding: 0.35rem 0.9rem !important;
}
[data-testid="stPopover"] button:hover { border-color: var(--brand-2) !important; }
/* Hide the built-in expand/collapse chevron so only the bold dots show */
[data-testid="stPopover"] button svg { display: none !important; }

/* The popover panel itself was inheriting Streamlit's default dark
   background + low-contrast text — pin both explicitly */
[data-testid="stPopoverBody"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    min-width: 220px;
}
[data-testid="stPopoverBody"] * { color: var(--ink) !important; }

/* Menu-item style rows ("Appearance" / "History" / "Back") inside the popover */
[data-testid="stPopoverBody"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    padding: 0.55rem 0.5rem !important;
}
[data-testid="stPopoverBody"] .stButton > button:hover { background: var(--brand-soft) !important; }

/* ---------- Big verdict hero — the ONE thing a judge should see first ---------- */
.verdict-hero {
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.9rem;
    margin: 0.7rem 0 1.4rem 0;
    border: 1px solid var(--border);
    box-shadow: 0 1px 0 rgba(255,255,255,0.6) inset, 0 14px 34px rgba(30,27,58,0.06);
}
.verdict-hero.hero-pass { background: linear-gradient(135deg, var(--clean-bg), transparent 70%); border-left: 8px solid var(--success); }
.verdict-hero.hero-review { background: linear-gradient(135deg, rgba(217,119,6,0.12), transparent 70%); border-left: 8px solid var(--warning); }
.verdict-hero.hero-fail { background: linear-gradient(135deg, var(--flag-bg), transparent 70%); border-left: 8px solid var(--danger); }

.vh-top { display: flex; align-items: flex-start; gap: 1.1rem; flex-wrap: wrap; }
.vh-icon-badge {
    width: 52px; height: 52px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    filter: drop-shadow(0 4px 10px rgba(30,27,58,0.16));
}
.hero-pass .vh-icon-badge { background: rgba(22,163,74,0.14); color: var(--success); }
.hero-review .vh-icon-badge { background: rgba(217,119,6,0.14); color: var(--warning); }
.hero-fail .vh-icon-badge { background: rgba(220,38,38,0.14); color: var(--danger); }
.vh-title-block { padding-top: 0.3rem; }
.vh-label { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.6rem; color: var(--ink); letter-spacing: -0.2px; line-height: 1.15; }
.vh-sub { color: var(--ink-muted); font-weight: 600; font-size: 0.95rem; margin-top: 0.15rem; }
.vh-score { margin-left: auto; text-align: right; padding-top: 0.3rem; }
.vh-score-num {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.9rem; color: var(--ink); line-height: 1.15;
    text-shadow: 0 1px 0 rgba(255,255,255,0.7), 0 6px 16px rgba(30,27,58,0.14);
}
.vh-score-cap { color: var(--ink-muted); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.3px; }
.vh-why, .vh-reco { margin-top: 0.9rem; font-size: 1.02rem; color: var(--ink); line-height: 1.5; }
.vh-reco { color: var(--ink-muted); }

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

/* ---------- Smart Progressive Disclosure — "Advanced Parameters" ----------
   Everything that isn't the drag-and-drop action (reference IDs, dates,
   location, model tuning) lives behind one quiet, low-contrast accordion so
   the default screen reads as a single clean dropzone, not a settings form. */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) [data-testid="stExpander"] {
    background: transparent !important;
    border: 1px dashed var(--border) !important;
    box-shadow: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) > div > [data-testid="stExpander"] > summary {
    padding: 0.7rem 0.9rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) [data-testid="stExpander"] summary p {
    font-weight: 600 !important; font-size: 0.9rem !important; color: var(--ink-muted) !important;
    letter-spacing: 0.1px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) [data-testid="stExpander"]:hover summary p {
    color: var(--brand-1) !important;
}
/* Nested sub-groups (Location) inside the drawer stay even quieter */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) [data-testid="stExpander"] [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    background: var(--surface-alt) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.adv-params-marker) [data-testid="stExpander"] [data-testid="stExpander"] summary p {
    font-size: 0.84rem !important;
}

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
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.4rem;
    color: var(--brand-1); white-space: nowrap; padding: 0 1.4rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.marquee-track span.dim { color: var(--brand-3); font-weight: 500; font-family: 'Inter', sans-serif; font-size: 1.05rem; }

/* ---------- Processing Theater — the multi-step "scanning" state shown
   while the forensics pipeline runs. This now replaces the ENTIRE hero view
   (headline, dropzone, marquee) rather than appearing alongside it, and
   uses a glassmorphic backdrop-blur card so it reads as a distinct,
   high-tech "command center" moment rather than a static wait. ---------- */
@keyframes pt-pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.15); opacity: 0.65; } }
@keyframes veda-fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.veda-view-fade { animation: veda-fade-in 0.45s ease both; }
/* A tall-ish stage so the theater card sits vertically centered in the
   primary viewport instead of hugging the top of the page */
div[class*="st-key-scan_stage"] { min-height: 62vh; display: flex; align-items: center; justify-content: center; }
.processing-theater {
    max-width: 640px; width: 100%; margin: 0 auto; text-align: center;
    background: color-mix(in srgb, var(--surface) 62%, transparent);
    backdrop-filter: blur(22px) saturate(160%); -webkit-backdrop-filter: blur(22px) saturate(160%);
    border: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
    border-radius: var(--radius-lg);
    padding: 2.3rem 2.5rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.5) inset, 0 30px 70px rgba(99,102,241,0.22), 0 0 0 1px rgba(99,102,241,0.06);
}
.pt-icon { font-size: 2.6rem; animation: pt-pulse 1.5s ease-in-out infinite; filter: drop-shadow(0 6px 14px rgba(139,92,246,0.4)); }
.pt-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.4rem; color: var(--ink); margin-top: 0.5rem; }
.pt-file { color: var(--ink-muted); font-size: 0.9rem; margin: 0.3rem 0 1.5rem 0; }
.pt-steps { text-align: left; display: flex; flex-direction: column; gap: 0.55rem; }
.pt-step {
    font-size: 0.94rem; padding: 0.55rem 0.9rem; border-radius: var(--radius-sm);
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--surface-alt); color: var(--ink-muted); border: 1px solid transparent;
    transition: all 0.25s ease;
}
.pt-step.done { color: var(--success); }
.pt-step.active {
    color: var(--brand-1); font-weight: 700; background: var(--brand-soft);
    border-color: rgba(99,102,241,0.3);
}

/* Style Streamlit's progress bar fill to match the brand gradient */
[data-testid="stProgress"] div[role="progressbar"] > div {
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important;
}

[data-testid="stMetricValue"] { color: var(--brand-1) !important; font-family: 'Inter', sans-serif !important; font-weight: 700 !important; font-size: 1.65rem !important; }
[data-testid="stMetricLabel"] { color: var(--ink-muted) !important; }
/* Technical/hash-like data reads as monospace for a "cybersecurity" feel,
   and is sized down so it doesn't threaten to overflow its container */
.mono-tech {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 0.85rem !important; color: var(--ink) !important;
}

hr, div[data-testid="stDivider"] { border-color: var(--border) !important; }

/* ---------- Pill-style navigation tabs (Explainability / File Forensics /
   etc.) — active tab reads as a solid toggle button, not a plain text link,
   so it's obvious this is an interactive dashboard. Targeted two ways: via
   data-baseweb (BaseWeb's own attributes) AND via the stable ARIA role +
   stTabs testid, since data-baseweb isn't guaranteed stable across
   Streamlit versions but role="tab" and data-testid="stTabs" are. ---------- */
[data-baseweb="tab-list"], [data-testid="stTabs"] [role="tablist"] {
    gap: 0.4rem !important; background: var(--surface-alt);
    padding: 0.4rem !important; border-radius: 999px !important; display: inline-flex !important;
    border-bottom: none !important; width: fit-content;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; background: transparent !important; }
button[data-baseweb="tab"], [data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px !important; padding: 0.5rem 1.15rem !important;
    color: var(--ink-muted) !important; font-weight: 600 !important; font-size: 0.92rem !important;
    background: transparent !important; border: none !important;
    transition: background 0.2s ease, color 0.2s ease; margin: 0 !important;
}
button[data-baseweb="tab"]:hover, [data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(99,102,241,0.09) !important; color: var(--brand-1) !important;
}
button[data-baseweb="tab"][aria-selected="true"], [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.32);
}
button[data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * { color: #ffffff !important; }

/* =============================================================================
   3D VISUAL SYSTEM — rotating scan-orb logo, holographic perspective floor,
   true perspective tilt on cards, and a 3D flip-in verdict badge. Pure CSS
   (no JS dependency, so it renders reliably inside Streamlit markdown).
   ============================================================================= */

/* ---------- Static 3D "Scan Orb" — replaces the flat emoji logo dot.
   No longer rotates/tumbles per request; keeps the tilted 3D pose it
   started from, just held still. ---------- */
.orb-3d-scene {
    width: 60px; height: 60px; flex-shrink: 0;
    perspective: 300px; perspective-origin: 50% 50%;
}
.orb-3d {
    position: relative; width: 100%; height: 100%;
    transform-style: preserve-3d;
    transform: rotateY(0deg) rotateX(18deg);
}
.orb-core {
    position: absolute; inset: 6px; border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    box-shadow: 0 0 22px rgba(99,102,241,0.55), inset -5px -5px 12px rgba(30,20,70,0.35),
                inset 4px 4px 10px rgba(255,255,255,0.45);
    display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
    transform: translateZ(13px);
}
.orb-ring {
    position: absolute; inset: 0; border-radius: 50%;
    border: 1.5px solid rgba(99,102,241,0.55);
    box-shadow: 0 0 10px rgba(99,102,241,0.25);
}
.orb-ring-1 { transform: rotateX(72deg) translateZ(0px); }
.orb-ring-2 { transform: rotateY(72deg) translateZ(0px); border-color: rgba(139,92,246,0.5); }
.orb-ring-3 { transform: rotateX(35deg) rotateY(35deg) translateZ(0px); border-color: rgba(167,139,250,0.4); }

/* ---------- Holographic perspective scan-floor behind the hub hero ---------- */
.scan-floor-wrap {
    position: relative; height: 0; overflow: visible;
}
.scan-floor {
    position: absolute; left: 50%; top: -0.4rem;
    width: 1100px; max-width: 160vw; height: 260px;
    transform: translateX(-50%) perspective(420px) rotateX(62deg);
    background-image:
        linear-gradient(var(--brand-3) 1px, transparent 1px),
        linear-gradient(90deg, var(--brand-3) 1px, transparent 1px);
    background-size: 44px 44px;
    opacity: 0.16;
    -webkit-mask-image: radial-gradient(ellipse 55% 100% at 50% 0%, black 35%, transparent 78%);
    mask-image: radial-gradient(ellipse 55% 100% at 50% 0%, black 35%, transparent 78%);
    animation: scan-floor-drift 4.5s linear infinite;
    pointer-events: none; z-index: 0;
}
@keyframes scan-floor-drift {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 0 44px, 44px 0; }
}

/* ---------- True perspective tilt on the central hub card ---------- */
.hub-perspective-wrap { perspective: 1400px; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hub-marker) {
    transform-style: preserve-3d;
    transform: perspective(1400px) rotateX(0.6deg);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hub-marker):hover {
    transform: perspective(1400px) rotateX(1.6deg) translateY(-4px) scale(1.004);
}

/* ---------- 3D bulge tilt on the forensic metric tiles ---------- */
div[class*="st-key-metric_card_"] { transform-style: preserve-3d; }
div[class*="st-key-metric_card_"]:hover {
    transform: perspective(700px) rotateX(6deg) translateY(-8px) scale(1.035);
}

/* ---------- 3D flip-in animation for the verdict hero icon badge ---------- */
@keyframes badge-flip-in {
    0%   { transform: perspective(400px) rotateY(-140deg) scale(0.6); opacity: 0; }
    60%  { transform: perspective(400px) rotateY(15deg) scale(1.05); opacity: 1; }
    100% { transform: perspective(400px) rotateY(0deg) scale(1); opacity: 1; }
}
.vh-icon-badge {
    animation: badge-flip-in 0.7s cubic-bezier(0.22,1,0.36,1) both;
    transform-style: preserve-3d;
    position: relative;
}
.vh-icon-badge::before {
    content: ""; position: absolute; inset: -6px; border-radius: 50%;
    border: 1.5px dashed currentColor; opacity: 0.35;
    animation: badge-halo-spin 6s linear infinite;
}
@keyframes badge-halo-spin {
    0%   { transform: perspective(300px) rotateX(60deg) rotateZ(0deg); }
    100% { transform: perspective(300px) rotateX(60deg) rotateZ(360deg); }
}
.verdict-hero { perspective: 900px; }

/* ---------- Subtle 3D depth on result cards (image/video + metric groups) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker) {
    transform-style: preserve-3d;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-marker):hover {
    transform: perspective(900px) rotateX(1.4deg) translateY(-4px);
}
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
if "is_scanning" not in st.session_state:
    st.session_state.is_scanning = False
if "use_case" not in st.session_state:
    st.session_state.use_case = "General / Personal Verification"
if "menu_view" not in st.session_state:
    st.session_state.menu_view = "root"

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
# 6c. PIXEL-LEVEL FORENSICS MODULE — frequency-domain, noise-residual &
# illumination analysis, plus embedded provenance (C2PA/Content Credentials)
# detection. Runs for BOTH images and videos. These are classical,
# model-free signal-processing checks that catch generation artifacts the
# deep-learning classifier and scene module can miss entirely:
#   1) GAN/diffusion upsampling leaves periodic spectral artifacts
#      (transposed-conv "checkerboarding") invisible to the naked eye
#   2) Real camera sensor noise is fairly uniform; AI output is often
#      unnaturally smooth, or patchy where regions were composited
#   3) Real exposure changes gradually even during fast motion; abrupt
#      unexplained lighting jumps are a common splice/generation seam
#   4) Embedded C2PA/Content Credentials metadata is a positive
#      corroborating signal when present (its absence is NOT suspicious —
#      most legitimate media still doesn't carry it)
# =============================================================================
def _radial_profile(magnitude_spectrum):
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = min(cy, cx)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    profile = tbin / nr
    return profile[:r_max] if r_max > 0 else profile

def analyze_spectral_artifacts(frame_gray) -> dict:
    """Real photos have a smooth, power-law-decaying radial frequency
    spectrum. GAN/diffusion upsampling tends to leave sharp, regularly
    spaced peaks in the mid-to-high frequency band. We measure how spiky
    that band is relative to its own local trend — scale-invariant across
    different images/exposures."""
    try:
        g = cv2.resize(frame_gray, (256, 256))
        f = np.fft.fft2(g)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-8)
        profile = _radial_profile(magnitude)
        if len(profile) < 12:
            return {"spectral_anomaly_score": 0.0, "spectral_flag": False}
        mid = profile[6:]
        if len(mid) < 6:
            return {"spectral_anomaly_score": 0.0, "spectral_flag": False}
        smoothed = np.convolve(mid, np.ones(5) / 5, mode="same")
        residual = mid - smoothed
        spikiness = float(np.std(residual) / (np.std(mid) + 1e-6))
        anomaly_score = float(np.clip(spikiness, 0.0, 1.0))
        return {"spectral_anomaly_score": round(anomaly_score, 3), "spectral_flag": anomaly_score > 0.55}
    except Exception:
        return {"spectral_anomaly_score": 0.0, "spectral_flag": False}

def analyze_noise_residual(frame_bgr) -> dict:
    """High-pass noise residual (frame minus a Gaussian-blurred version of
    itself), measured in an 8x8 block grid. Flags content that's either
    unnaturally smooth/denoised overall (common in AI-generated output) or
    unnaturally patchy across regions (a compositing/splicing tell)."""
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray - blurred
        h, w = residual.shape
        gh, gw = max(1, h // 8), max(1, w // 8)
        block_vars = []
        for i in range(0, h - gh + 1, gh):
            for j in range(0, w - gw + 1, gw):
                block = residual[i:i + gh, j:j + gw]
                if block.size > 0:
                    block_vars.append(float(np.var(block)))
        if not block_vars:
            return {"noise_mean_var": 0.0, "noise_cv": 0.0, "too_smooth_flag": False, "patchy_flag": False}
        mean_var = float(np.mean(block_vars))
        cv_ratio = float(np.std(block_vars) / (mean_var + 1e-6))
        return {
            "noise_mean_var": round(mean_var, 3), "noise_cv": round(cv_ratio, 3),
            "too_smooth_flag": mean_var < 1.2, "patchy_flag": cv_ratio > 1.6,
        }
    except Exception:
        return {"noise_mean_var": 0.0, "noise_cv": 0.0, "too_smooth_flag": False, "patchy_flag": False}

def analyze_illumination_consistency(frames_bgr) -> dict:
    """Frame-to-frame mean-luminance delta across sampled frames. An abrupt,
    large jump with no corresponding scene motion is a common seam in
    AI-generated or spliced video."""
    if len(frames_bgr) < 3:
        return {"illum_series": [], "illum_jump_flag": False, "max_illum_jump": 0.0}
    lum = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames_bgr]
    deltas = [abs(lum[i] - lum[i - 1]) for i in range(1, len(lum))]
    max_jump = float(np.max(deltas)) if deltas else 0.0
    return {"illum_series": [round(v, 2) for v in lum], "illum_jump_flag": max_jump > 35.0, "max_illum_jump": round(max_jump, 2)}

def detect_content_credentials(file_path: str) -> dict:
    """Scans raw file bytes for embedded C2PA / Content Credentials
    provenance markers — the open standard cameras, Adobe, and some AI
    tools use to attest capture/edit history. Presence is a corroborating
    positive signal; absence is NOT treated as suspicious on its own."""
    try:
        with open(file_path, "rb") as f:
            data = f.read(2_000_000)
        markers = [b"c2pa", b"C2PA", b"jumbf", b"JUMBF", b"Content Credentials", b"urn:uuid:c2pa"]
        found = any(m in data for m in markers)
        return {"has_content_credentials": found}
    except Exception:
        return {"has_content_credentials": False}

def analyze_pixel_forensics(file_path: str, is_image: bool) -> dict:
    """Unified entry point run for both images and videos."""
    frames = []
    if is_image:
        img = cv2.imread(file_path)
        if img is not None:
            frames = [img]
    else:
        cap = cv2.VideoCapture(file_path)
        count = 0
        while cap.isOpened() and count < 8:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        cap.release()

    empty = {
        "spectral_anomaly_score": 0.0, "spectral_flag": False,
        "noise_mean_var": 0.0, "noise_cv": 0.0, "too_smooth_flag": False, "patchy_flag": False,
        "illum_series": [], "illum_jump_flag": False, "max_illum_jump": 0.0,
        "has_content_credentials": False,
    }
    if not frames:
        return empty

    spectral_scores, noise_results = [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        spectral_scores.append(analyze_spectral_artifacts(gray))
        noise_results.append(analyze_noise_residual(f))

    avg_spectral = float(np.mean([s["spectral_anomaly_score"] for s in spectral_scores]))
    avg_noise_var = float(np.mean([n["noise_mean_var"] for n in noise_results]))
    avg_noise_cv = float(np.mean([n["noise_cv"] for n in noise_results]))
    illum_result = analyze_illumination_consistency(frames) if not is_image else empty
    provenance = detect_content_credentials(file_path)

    return {
        "spectral_anomaly_score": round(avg_spectral, 3), "spectral_flag": avg_spectral > 0.55,
        "noise_mean_var": round(avg_noise_var, 3), "noise_cv": round(avg_noise_cv, 3),
        "too_smooth_flag": avg_noise_var < 1.2, "patchy_flag": avg_noise_cv > 1.6,
        "illum_series": illum_result["illum_series"], "illum_jump_flag": illum_result["illum_jump_flag"],
        "max_illum_jump": illum_result["max_illum_jump"],
        "has_content_credentials": provenance["has_content_credentials"],
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
                                  ela_score, duplicate_flag, scene_result=None, pixel_result=None):
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
            d = 35; score -= d
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

    # --- Pixel-level forensics: frequency-domain GAN/diffusion artifacts,
    # noise-residual smoothness/patchiness, and illumination consistency ---
    if pixel_result:
        if pixel_result.get("spectral_flag"):
            d = 15; score -= d
            reasons.append(f"Frequency-domain analysis detected periodic upsampling artifacts "
                            f"(spectral anomaly score {pixel_result.get('spectral_anomaly_score', 0):.2f}) "
                            f"consistent with GAN/diffusion-generated content")
            deductions.append(("Spectral/Frequency Artifacts", d))
        if pixel_result.get("too_smooth_flag"):
            d = 10; score -= d
            reasons.append("Noise residual is abnormally smooth for camera-captured media — "
                            "consistent with AI generation or heavy denoising")
            deductions.append(("Abnormally Smooth Noise Floor", d))
        if pixel_result.get("patchy_flag"):
            d = 12; score -= d
            reasons.append("Noise residual is inconsistent across regions of the frame — "
                            "a possible sign of image splicing/compositing")
            deductions.append(("Patchy Noise Residual", d))
        if pixel_result.get("illum_jump_flag"):
            d = 8; score -= d
            reasons.append(f"Abrupt frame-to-frame lighting shift ({pixel_result.get('max_illum_jump', 0):.1f} "
                            f"luminance jump) inconsistent with natural camera exposure changes")
            deductions.append(("Illumination Discontinuity", d))

    final_score = max(0.0, round(score, 1))
    # Verdict thresholds: >80 → Real/Authentic, 70–80 → Manual Review,
    # <70 → Fake/High Fraud Risk.
    if final_score > 80:
        status = "MANUAL REVIEW REQUIRED" if force_manual_review else "PASSED (AUTHENTIC)"
    elif final_score >= 70:
        status = "MANUAL REVIEW REQUIRED"
    else:
        status = "REJECTED (HIGH FRAUD RISK)"
    return final_score, status, reasons, deductions

# =============================================================================
# 8. UI CHART HELPERS
# =============================================================================
def render_gauge(score: float):
    color = "#10b981" if score > 80 else ("#f59e0b" if score >= 70 else "#ef4444")
    # Plotly's default text color is a faint gray that washes out on our
    # theme's backgrounds — pin it explicitly, and keep it theme-aware so it
    # stays legible in both Light and Dark mode.
    ink_color = "#EDEBFA" if st.session_state.get("theme_mode") == "Dark" else "#1E1B3A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 64, "color": ink_color, "family": "Fraunces, serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 14, "color": ink_color}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [0, 70], "color": "#fee2e2"}, {"range": [70, 80], "color": "#fef3c7"}, {"range": [80, 100], "color": "#dcfce7"}],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.9, "value": score},
        },
        title={"text": "AUTHENTICITY SCORE", "font": {"size": 16, "color": ink_color}},
    ))
    fig.update_layout(height=420, margin=dict(l=30, r=30, t=70, b=20), paper_bgcolor="rgba(0,0,0,0)")
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

# =============================================================================
# 8b. PDF REPORT — branded "VeriLens AI" report with charts + conclusion
# =============================================================================
def _pdf_fig_to_buf(fig):
    """Render a matplotlib figure to an in-memory PNG buffer for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _pdf_score_chart(score: float):
    color = "#16A34A" if score > 80 else ("#D97706" if score >= 70 else "#DC2626")
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh([0], [100], color="#E7E6F5", height=0.55, zorder=0)
    ax.barh([0], [min(score, 100)], color=color, height=0.55, zorder=1)
    ax.axvline(70, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.axvline(80, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.text(70, 0.42, "70", ha="center", fontsize=8, color="#6B7280")
    ax.text(80, 0.42, "80", ha="center", fontsize=8, color="#6B7280")
    ax.text(min(score, 100), 0, f"  {score}/100", va="center", fontsize=13, fontweight="bold", color="#1E1B3A")
    ax.set_xlim(0, 105); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([]); ax.set_xlabel("Authenticity Score  (0–70 Fake · 70–80 Manual Review · 80–100 Real)", fontsize=8.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _pdf_fig_to_buf(fig)

def _pdf_deduction_chart(deductions):
    if not deductions:
        return None
    labels = [d[0] for d in deductions]
    values = [d[1] for d in deductions]
    fig, ax = plt.subplots(figsize=(6.2, max(1.6, 0.42 * len(labels))))
    ax.barh(labels, values, color="#DC2626")
    ax.invert_yaxis()
    ax.set_xlabel("Points Deducted", fontsize=9)
    ax.tick_params(labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _pdf_fig_to_buf(fig)

def _pdf_timeline_chart(frame_data):
    if not frame_data:
        return None
    xs = [f[0] for f in frame_data]
    ys = [f[1] for f in frame_data]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(xs, ys, marker="o", markersize=3, color="#6366F1", linewidth=1.6)
    ax.axhline(0.5, color="#D97706", linestyle="--", linewidth=1, label="Decision threshold")
    ax.axhline(0.6, color="#DC2626", linestyle=":", linewidth=1, label="Suspicious frame threshold")
    ax.set_xlabel("Timestamp (s)", fontsize=9); ax.set_ylabel("Deepfake probability", fontsize=9)
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _pdf_fig_to_buf(fig)

def _pdf_header_footer(canvas, doc):
    canvas.saveState()
    page_w, page_h = doc.pagesize
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(rl_colors.HexColor("#6D28D9"))
    canvas.drawString(0.6 * inch, page_h - 0.45 * inch, "VeriLens AI")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(rl_colors.HexColor("#9CA3AF"))
    canvas.drawRightString(page_w - 0.6 * inch, page_h - 0.45 * inch, "Deepfake & Authenticity Verification Report")
    canvas.setStrokeColor(rl_colors.HexColor("#E7E6F5"))
    canvas.line(0.6 * inch, page_h - 0.55 * inch, page_w - 0.6 * inch, page_h - 0.55 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(rl_colors.HexColor("#9CA3AF"))
    canvas.drawString(0.6 * inch, 0.4 * inch, "Generated by VeriLens AI — automated analysis, not a substitute for human judgment")
    canvas.drawRightString(page_w - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()

def generate_pdf_report(r: dict) -> bytes:
    """Builds a full branded VeriLens AI PDF report for a single result: verdict,
    score, forensic breakdown charts, flagged indicators, and a written conclusion."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.85 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("VLTitle", parent=styles["Title"], textColor=rl_colors.HexColor("#4338CA"),
                                  fontSize=25, spaceAfter=2, alignment=0)
    brand_sub = ParagraphStyle("VLBrandSub", parent=styles["Normal"], textColor=rl_colors.HexColor("#6B7280"),
                                fontSize=10.5, spaceAfter=14)
    h2 = ParagraphStyle("VLH2", parent=styles["Heading2"], textColor=rl_colors.HexColor("#1E1B3A"),
                         fontSize=13.5, spaceBefore=16, spaceAfter=6)
    body = ParagraphStyle("VLBody", parent=styles["Normal"], fontSize=10.2, leading=15,
                           textColor=rl_colors.HexColor("#27253F"))
    flag_style = ParagraphStyle("VLFlag", parent=body, leftIndent=10, spaceAfter=3)

    status = r["status"]
    if status.startswith("PASSED"):
        verdict_label, verdict_hex = "REAL / AUTHENTIC", "#16A34A"
    elif status.startswith("MANUAL"):
        verdict_label, verdict_hex = "MANUAL REVIEW REQUIRED", "#D97706"
    else:
        verdict_label, verdict_hex = "FAKE / HIGH FRAUD RISK", "#DC2626"
    verdict_style = ParagraphStyle("VLVerdict", parent=styles["Heading1"], textColor=rl_colors.HexColor(verdict_hex),
                                    fontSize=19, spaceAfter=4)

    story = []
    story.append(Paragraph("VeriLens AI", title_style))
    story.append(Paragraph("Universal Deepfake &amp; Media Authenticity Verification Report", brand_sub))
    story.append(HRFlowable(width="100%", color=rl_colors.HexColor("#E7E6F5"), thickness=1))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Verdict: {verdict_label}", verdict_style))
    story.append(Paragraph(
        f"Authenticity Score: <b>{r['score']} / 100</b> &nbsp;·&nbsp; AI Model Prediction: <b>{r['pred_label']}</b>",
        body))
    story.append(Spacer(1, 10))

    current_labels = USE_CASES[r["use_case"]]
    details = [
        [current_labels["id_label"], r["ref_id"] or "—"],
        [current_labels["name_label"], r["subject_name"] or "—"],
        ["File Name", r["file_name"]],
        ["Media Type", "Image" if r["is_image"] else "Video"],
        ["Use Case", r["use_case"]],
        ["Analyzed On", r["timestamp"]],
        ["SHA-256", r["sha256"]],
    ]
    t = Table(details, colWidths=[1.7 * inch, 4.7 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("TEXTCOLOR", (0, 0), (0, -1), rl_colors.HexColor("#6B7280")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E7E6F5")),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Authenticity Score Overview", h2))
    story.append(RLImage(_pdf_score_chart(r["score"]), width=6 * inch, height=1.6 * inch))

    story.append(Paragraph("Risk Indicators &amp; Score Deductions", h2))
    if r["flags"]:
        for flag in r["flags"]:
            story.append(Paragraph(f"• {flag}", flag_style))
        ded_buf = _pdf_deduction_chart(r["deductions"])
        if ded_buf:
            story.append(Spacer(1, 6))
            n = max(1, len(r["deductions"]))
            story.append(RLImage(ded_buf, width=6.2 * inch, height=min(4.5, 0.42 * n + 0.8) * inch))
    else:
        story.append(Paragraph("No suspicious manipulation indicators were detected.", body))

    if not r["is_image"] and r.get("frame_data"):
        tl_buf = _pdf_timeline_chart(r["frame_data"])
        if tl_buf:
            story.append(Paragraph("Frame-Level Deepfake Probability", h2))
            story.append(RLImage(tl_buf, width=6.2 * inch, height=2.6 * inch))

    scene = r.get("scene_result")
    if not r["is_image"] and scene:
        story.append(Paragraph("Scene Plausibility", h2))
        if scene.get("flags"):
            for flag in scene["flags"]:
                story.append(Paragraph(f"• {flag}", flag_style))
        else:
            story.append(Paragraph("No unmotivated pyrotechnics, plate inconsistency, or motion-physics anomalies detected.", body))

    px = r.get("pixel_result")
    if px:
        story.append(Paragraph("Advanced Pixel-Level Forensics", h2))
        px_flags = []
        if px.get("spectral_flag"):
            px_flags.append(f"Frequency-domain analysis found periodic upsampling artifacts "
                             f"(anomaly score {px.get('spectral_anomaly_score', 0):.2f}) consistent with "
                             f"GAN/diffusion generation.")
        if px.get("too_smooth_flag"):
            px_flags.append(f"Noise floor is abnormally smooth (variance {px.get('noise_mean_var', 0):.2f}) — "
                             f"consistent with AI generation or heavy denoising.")
        if px.get("patchy_flag"):
            px_flags.append(f"Noise residual is inconsistent across the frame "
                             f"(coefficient of variation {px.get('noise_cv', 0):.2f}) — possible splicing.")
        if px.get("illum_jump_flag"):
            px_flags.append(f"Abrupt lighting jump detected ({px.get('max_illum_jump', 0):.1f} luminance "
                             f"units) inconsistent with natural camera exposure.")
        if px_flags:
            for flag in px_flags:
                story.append(Paragraph(f"• {flag}", flag_style))
        else:
            story.append(Paragraph("No frequency-domain, noise-residual, or illumination anomalies detected.", body))
        story.append(Paragraph(
            f"Spectral anomaly score: <b>{px.get('spectral_anomaly_score', 0):.2f}</b> &nbsp;·&nbsp; "
            f"Noise floor variance: <b>{px.get('noise_mean_var', 0):.2f}</b> &nbsp;·&nbsp; "
            f"Noise patchiness (CV): <b>{px.get('noise_cv', 0):.2f}</b> &nbsp;·&nbsp; "
            f"Content Credentials (C2PA): <b>{'Detected' if px.get('has_content_credentials') else 'Not found'}</b>",
            body))

    story.append(PageBreak())
    story.append(Paragraph("Conclusion", h2))
    media_word = "image" if r["is_image"] else "video"
    if status.startswith("PASSED"):
        conclusion = (
            f"Based on a comprehensive multi-layer forensic analysis, this {media_word} achieved an authenticity "
            f"score of {r['score']}/100, placing it above the 80-point authenticity threshold. No significant "
            f"manipulation indicators were identified across the AI classification, file forensics, and context "
            f"checks performed. VeriLens AI classifies this content as <b>REAL / AUTHENTIC</b>. No further action "
            f"is required, though standard verification practices are still recommended for high-stakes decisions."
        )
    elif status.startswith("MANUAL"):
        conclusion = (
            f"This {media_word} received an authenticity score of {r['score']}/100, which falls within the "
            f"70–80 point range. While the evidence gathered was not conclusive enough to classify this content "
            f"as clearly real or clearly fake, one or more risk indicators listed above warrant a closer look. "
            f"VeriLens AI classifies this content as <b>MANUAL REVIEW REQUIRED</b>. A qualified human reviewer "
            f"should examine the flagged indicators before this file is approved, published, or otherwise acted upon."
        )
    else:
        conclusion = (
            f"This {media_word} received an authenticity score of {r['score']}/100, falling below the 70-point "
            f"authenticity threshold. Multiple strong risk indicators were identified during analysis, "
            f"significantly reducing confidence in the content's authenticity. VeriLens AI classifies this "
            f"content as <b>FAKE / HIGH FRAUD RISK</b>. This content should not be approved, published, or relied "
            f"upon without further independent verification."
        )
    story.append(Paragraph(conclusion, body))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        f"<i>Report generated by VeriLens AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</i>",
        ParagraphStyle("VLFoot", parent=body, fontSize=8.7, textColor=rl_colors.HexColor("#9CA3AF"))))

    doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    buf.seek(0)
    return buf.getvalue()

def verdict_banner(status: str):
    if status.startswith("PASSED"):
        st.markdown(f'<div class="verdict-pass">✅ {status} — No significant manipulation indicators found.</div>', unsafe_allow_html=True)
    elif status.startswith("MANUAL"):
        st.markdown(f'<div class="verdict-review">🟠 {status} — Some risk indicators present, human review recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-fail">🔴 {status} — Multiple strong risk indicators detected.</div>', unsafe_allow_html=True)

def score_based_classification(score: float) -> str:
    """
    Display-only label for the "AI Classification" tile, derived from the
    already-computed final authenticity score rather than the raw model
    pred_label — so the tile agrees with the score a viewer sees right next
    to it. This does not change calculate_authenticity_score or pred_label
    itself, only what text this one tile shows.
    """
    if score < 70:
        return "FAKE"
    elif score <= 80:
        return "MANUAL REVIEW REQUIRED"
    else:
        return "REAL"

def verdict_hero(r: dict):
    """
    The single most important element on the results screen: a large,
    color-coded verdict with the top reason, risk level, score, and a
    plain-language recommendation — everything else on the page is detail
    a viewer can dig into afterward, but this is what they should see first.
    """
    ICONS = {
        # Shield + checkmark — a deliberate "verified" icon, not a plain circle
        "pass": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M12 2.5l7.5 3v5.4c0 5-3.2 9.1-7.5 10.6-4.3-1.5-7.5-5.6-7.5-10.6V5.5l7.5-3z" '
                'fill="currentColor" fill-opacity="0.16" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>'
                '<path d="M8.3 12.3l2.5 2.5 4.9-5.2" stroke="currentColor" stroke-width="1.9" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>',
        # Warning triangle + exclamation — a sharp, unambiguous caution icon
        "review": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
                  '<path d="M12 3.2L21.3 20H2.7L12 3.2z" fill="currentColor" fill-opacity="0.16" '
                  'stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>'
                  '<path d="M12 9.3V14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                  '<circle cx="12" cy="16.9" r="1.15" fill="currentColor"/></svg>',
        # Alert circle — clear high-risk marker distinct from the other two
        "fail": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="12" cy="12" r="9.3" fill="currentColor" fill-opacity="0.16" stroke="currentColor" stroke-width="1.6"/>'
                '<path d="M12 7.3V13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                '<circle cx="12" cy="16.2" r="1.15" fill="currentColor"/></svg>',
    }

    status = r["status"]
    if status.startswith("PASSED"):
        theme = {
            "cls": "hero-pass", "icon": ICONS["pass"], "label": "AUTHENTIC", "risk": "Low Risk",
            "reco": "No further action needed — content appears genuine.",
        }
    elif status.startswith("MANUAL"):
        theme = {
            "cls": "hero-review", "icon": ICONS["review"], "label": "MANUAL REVIEW REQUIRED", "risk": "Medium Risk",
            "reco": "Some indicators are ambiguous — a human reviewer should confirm before acting on this.",
        }
    else:
        theme = {
            "cls": "hero-fail", "icon": ICONS["fail"], "label": "DEEPFAKE / HIGH FRAUD RISK", "risk": "High Risk",
            "reco": "Strong manipulation signals detected — treat as unverified; do not approve or publish.",
        }

    top_reason = r["flags"][0] if r["flags"] else "No significant manipulation indicators were found."

    st.markdown(f"""
        <div class="verdict-hero {theme['cls']}">
            <div class="vh-top">
                <span class="vh-icon-badge">{theme['icon']}</span>
                <div class="vh-title-block">
                    <div class="vh-label">{theme['label']}</div>
                    <div class="vh-sub">{theme['risk']}</div>
                </div>
                <div class="vh-score">
                    <div class="vh-score-num">{r['score']}</div>
                    <div class="vh-score-cap">AUTHENTICITY SCORE</div>
                </div>
            </div>
            <div class="vh-why">🔍 <b>Why:</b> {top_reason}</div>
            <div class="vh-reco">💡 <b>Recommendation:</b> {theme['reco']}</div>
        </div>
    """, unsafe_allow_html=True)

def render_appearance_history_icons():
    """
    A single "⋮" button replacing the two separate icon popovers — clicking it
    reveals a small menu with Appearance and History as selectable rows
    (Appearance first, per request), each drilling into its own view with a
    Back row to return, the same pattern used in chat apps like WhatsApp.
    """
    # Real marker (not a wrapping div, which Streamlit doesn't actually nest
    # around the next widget) — the CSS :has() rule uses this to right-align
    # the trigger within its column.
    st.markdown('<span class="topbar-icon-marker"></span>', unsafe_allow_html=True)
    with st.popover("⋮"):
        if st.session_state.menu_view == "root":
            if st.button("🎨  Appearance", key="menu_open_appearance", use_container_width=True):
                st.session_state.menu_view = "appearance"
                st.rerun()
            hist_count = len(st.session_state.history)
            if st.button(f"🕒  History ({hist_count})" if hist_count else "🕒  History",
                         key="menu_open_history", use_container_width=True):
                st.session_state.menu_view = "history"
                st.rerun()

        elif st.session_state.menu_view == "appearance":
            if st.button("‹  Back", key="menu_back_from_appearance", use_container_width=True):
                st.session_state.menu_view = "root"
                st.rerun()
            st.caption("Appearance")
            st.radio(
                "Theme", ["Light", "Dark"], key="theme_mode",
                label_visibility="collapsed",
            )

        else:  # "history"
            if st.button("‹  Back", key="menu_back_from_history", use_container_width=True):
                st.session_state.menu_view = "root"
                st.rerun()
            st.caption("Session History")
            if st.session_state.history:
                hist_df = pd.DataFrame([{"Time": h["timestamp"], "File": h["file_name"], "Status": h["status"]} for h in st.session_state.history])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
                if st.button("🗑️ Clear History", use_container_width=True, key="clear_history_btn"):
                    st.session_state.history = []
                    st.session_state.pop("last_results", None)
                    st.rerun()
            else:
                st.caption("No verifications yet this session.")

# =============================================================================
# 9. TWO-STATE UI IMPLEMENTATION
# =============================================================================
if not st.session_state.is_analyzed:
    # --- Topbar shown in both the hub view and the scanning view ---
    topbar_l, topbar_c, topbar_r = st.columns([1, 10, 1])
    with topbar_c:
        st.markdown("""
            <div class="veda-topbar">
                <div class="veda-topbar-row">
                    <div class="logo-dot orb-3d-scene"><div class="orb-3d"><div class="orb-ring orb-ring-1"></div><div class="orb-ring orb-ring-2"></div><div class="orb-ring orb-ring-3"></div><div class="orb-core">✨</div></div></div>
                    <div class="logo-text">VeriLens AI</div>
                </div>
                <div class="logo-sub">Insurance Claim Deepfake &amp; Authenticity Verifier</div>
            </div>
        """, unsafe_allow_html=True)
    with topbar_r:
        render_appearance_history_icons()

    if st.session_state.is_scanning:
        # --- STATE 1b: DEDICATED SCANNING VIEW ---
        # The hero headline, dropzone, and marquee are not merely covered by
        # this card — they are not rendered at all in this branch. This is a
        # clean, separate view driven by session state, not the hub view with
        # something swapped inside it, which is what caused the old bug
        # (headline/marquee still rendering above/below the theater).
        pending_files = st.session_state.get("pending_uploaded_files", [])
        ps = st.session_state.get("pending_settings", {})
        ref_id = ps.get("ref_id", "REF-2026-0001")
        subject_name = ps.get("subject_name", "")
        incident_date = ps.get("incident_date", date.today())
        reported_weather = ps.get("reported_weather", "Not specified")
        lat = ps.get("lat", 28.6139)
        lon = ps.get("lon", 77.2090)
        owm_api_key = ps.get("owm_api_key", "")
        max_frames = ps.get("max_frames", 20)
        ela_quality = ps.get("ela_quality", 90)

        # These three labels map to genuine stages of the pipeline below —
        # each message appears exactly when that real work is happening, not
        # as a fake animation layered on top.
        STEP_LABELS = [
            "🔍 Extracting metadata & cryptographic hashes...",
            "🧬 Running multi-frame biological and pixel-level analysis...",
            "🎬 Checking scene plausibility and duplicate logs...",
        ]

        def process_single_file(uploaded_file, progress_cb=None):
            def _tick(step_idx):
                if progress_cb:
                    progress_cb(step_idx)

            file_ext = uploaded_file.name.split(".")[-1].lower()
            is_image = file_ext in ["jpg", "jpeg", "png", "webp"]
            temp_path = f"temp_{int(time.time()*1000)}_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # --- Step 1: metadata + cryptographic hashes ---
            _tick(0)
            if is_image:
                metadata_res = extract_image_metadata(temp_path)
            else:
                metadata_res = extract_video_metadata(temp_path)
            file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            phash = average_hash(temp_path) if is_image else ""

            # --- Step 2: deepfake model inference + pixel-level (ELA) analysis ---
            _tick(1)
            if is_image:
                fake_prob, real_prob, pred_label = analyze_image_deepfake(temp_path)
                suspicious_ts, frame_data = [], []
                ela_image, ela_score = compute_ela_image(temp_path, quality=ela_quality)
                scene_result = None
            else:
                fake_prob, real_prob, pred_label, suspicious_ts, frame_data = analyze_video_deepfake(temp_path, max_frames=max_frames)
                ela_image, ela_score = None, 0.0
                scene_result = None

            # --- Step 3: scene plausibility, pixel forensics, context, and duplicate checks ---
            _tick(2)
            if not is_image:
                scene_result = analyze_scene_plausibility(temp_path, max_frames=max_frames)
            pixel_result = analyze_pixel_forensics(temp_path, is_image)
            weather_res = (check_weather_context(lat, lon, str(incident_date), reported_weather, owm_api_key)
                           if reported_weather != "Not specified" else {"matched": True, "api_weather": "N/A", "note": "No claim to verify"})
            fraud_ring_res = check_fraud_ring(temp_path, is_image=is_image)
            duplicate_flag = False

            for past in st.session_state.history:
                if past.get("sha256") == file_hash or (phash and past.get("phash") and hamming_distance(phash, past["phash"]) <= 4):
                    duplicate_flag = True
                    break

            score, status, flags, deductions = calculate_authenticity_score(
                fake_prob, pred_label, metadata_res, weather_res, fraud_ring_res, ela_score, duplicate_flag,
                scene_result, pixel_result
            )
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "file_name": uploaded_file.name,
                "use_case": st.session_state.use_case, "ref_id": ref_id, "subject_name": subject_name,
                "is_image": is_image, "temp_path": temp_path, "metadata": metadata_res, "fake_prob": fake_prob,
                "real_prob": real_prob, "pred_label": pred_label, "suspicious_ts": suspicious_ts,
                "frame_data": frame_data, "weather": weather_res, "fraud_ring": fraud_ring_res,
                "ela_score": ela_score, "duplicate_flag": duplicate_flag, "sha256": file_hash, "phash": phash,
                "score": score, "status": status, "flags": flags, "deductions": deductions, "ela_image": ela_image,
                "scene_result": scene_result, "pixel_result": pixel_result,
            }

        results = []
        total_files = len(pending_files)

        # A real Streamlit container (key= gives it a stable, directly
        # targetable CSS class) so the theater card can be vertically
        # centered in the primary viewport via min-height + flex.
        with st.container(key="scan_stage"):
            stage_l, stage_c, stage_r = st.columns([1, 5, 1])
            with stage_c:
                theater = st.empty()

        for i, uf in enumerate(pending_files):

            def _render_theater(step_idx, uf=uf, i=i):
                rows = ""
                for s_idx, label in enumerate(STEP_LABELS):
                    if s_idx < step_idx:
                        cls, icon = "done", "✅"
                    elif s_idx == step_idx:
                        cls, icon = "active", "⏳"
                    else:
                        cls, icon = "", "○"
                    rows += f'<div class="pt-step {cls}">{icon} {label}</div>'
                theater.markdown(f"""
                    <div class="processing-theater veda-view-fade">
                        <div class="pt-icon">✨</div>
                        <div class="pt-title">Running Multi-Layer Verification</div>
                        <div class="pt-file">📄 {uf.name} · File {i+1} of {total_files}</div>
                        <div class="pt-steps">{rows}</div>
                    </div>
                """, unsafe_allow_html=True)

            def _progress_cb(step_idx):
                _render_theater(step_idx)
                time.sleep(0.35)  # brief pacing so each stage is actually readable

            rec = process_single_file(uf, progress_cb=_progress_cb)
            results.append(rec)
            st.session_state.history.append(rec)

        st.session_state["last_results"] = results
        st.session_state.is_analyzed = True
        st.session_state.is_scanning = False
        st.session_state.pop("pending_uploaded_files", None)
        st.session_state.pop("pending_settings", None)
        st.rerun()

    else:
        # --- STATE 1a: NORMAL HUB VIEW (Gemini-style centered panel) ---
        st.markdown("""
            <div class="veda-hub-wrap veda-view-fade">
                <span class="veda-eyebrow"><span class="dot"></span> FORENSICS ENGINE ONLINE</span>
                <h1>Unyielding Authenticity for Digital Media.</h1>
                <p class="sub">Drop an image or video below. Metadata, pixel-level, deepfake-model, and scene-physics
                forensics run automatically — no setup required.</p>
            </div>
            <div class="scan-floor-wrap"><div class="scan-floor"></div></div>
        """, unsafe_allow_html=True)

        # Centered hub card — marker span makes the wrapping container
        # CSS-targetable.
        _hub_l, _hub_c, _hub_r = st.columns([1, 5, 1])
        with _hub_c:
            hub_placeholder = st.empty()

        with hub_placeholder.container():
            with st.container(border=True):
                st.markdown('<span class="hub-marker"></span>', unsafe_allow_html=True)
                st.markdown('<p style="font-weight:600; color:var(--ink-muted); font-size:0.82rem; '
                            'letter-spacing:0.3px; margin:0.9rem 0 0.5rem 0.4rem;">VERIFICATION USE CASE</p>',
                            unsafe_allow_html=True)
                st.radio(
                    "Verification Use Case", list(USE_CASES.keys()), key="use_case",
                    horizontal=True, label_visibility="collapsed",
                )

                # --- Smart progressive disclosure: every optional field the
                # sidebar used to hold now lives inside ONE quiet, collapsed
                # drawer, so the default screen is just the use-case chips
                # and the ask-bar below. ---
                st.markdown('<span class="adv-params-marker"></span>', unsafe_allow_html=True)
                labels = USE_CASES[st.session_state.use_case]
                with st.expander("⚙️  Advanced Parameters — reference ID, date, location & model tuning (optional)"):
                    ref_id = st.text_input(labels["id_label"], value="REF-2026-0001")
                    subject_name = st.text_input(labels["name_label"], value="")
                    incident_date = st.date_input("Content / Incident Date", value=date.today())
                    reported_weather = st.selectbox(
                        "Reported Environmental Context (optional)",
                        ["Not specified", "Clear", "Rain", "Snow", "Clouds", "Thunderstorm"],
                    )

                    with st.expander("📍 Location (for context check)"):
                        lat = st.number_input("Latitude", value=28.6139, format="%.4f")
                        lon = st.number_input("Longitude", value=77.2090, format="%.4f")
                        owm_api_key = st.text_input("OpenWeatherMap API Key", type="password")

                    with st.expander("🎚️ Advanced Settings"):
                        max_frames = st.slider("Video frames to sample", 5, 40, 20)
                        ela_quality = st.slider("ELA re-compression quality", 70, 95, 90)
                        show_ela_image = st.checkbox("Show ELA visualization", value=True)

                # --- The signature moment: a slim, rounded "ask-bar" — attach
                # pill on the left, circular send button on the right — the
                # same shape language as a Gemini-style chat composer. ---
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
                st.markdown('<span class="dropzone-marker"></span>', unsafe_allow_html=True)
                bar_up, bar_send = st.columns([11, 1.6])
                with bar_up:
                    uploaded_files = st.file_uploader(
                        "Upload Evidence (image or video, multiple allowed)",
                        type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "webp"],
                        accept_multiple_files=True,
                        label_visibility="collapsed",
                    )
                with bar_send:
                    st.markdown('<div style="height:3px;"></div>', unsafe_allow_html=True)
                    run_btn = st.button(
                        "➤", type="primary", key="gemini_send_btn",
                        help="Run Full Verification",
                    )
                st.markdown(
                    '<p class="gemini-bar-hint">Attach files, then tap ➤ to run the full forensic verification '
                    '&nbsp;·&nbsp; MP4, MOV, AVI, MKV, JPG, PNG, WEBP &nbsp;·&nbsp; up to 200MB per file</p>',
                    unsafe_allow_html=True,
                )

        st.markdown("""
            <div style='text-align:center; margin-top: 1.6rem;' class="veda-view-fade">
                <span class="badge badge-blue">MULTI-LAYER FORENSICS</span>
                <span class="badge badge-blue">IMAGE + VIDEO</span>
                <span class="badge badge-blue">BATCH READY</span>
            </div>
        """, unsafe_allow_html=True)
        st.caption("VeriLens AI · Multi-layer verification supports human review and is not a sole basis for legal or financial decisions.")

        # --- "Who We Are" scrolling brand marquee (VedaAI-style). ---
        _marquee_item = (
            '<span>✨ VeriLens AI</span><span class="dim">Trust every pixel, verify every frame</span>'
        )
        st.markdown(f"""
            <div class="marquee-band veda-view-fade">
                <div class="marquee-track">
                    {_marquee_item * 4}
                    {_marquee_item * 4}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Kick off the scanning view the moment "run" is pressed with files
        # attached — stash everything the pipeline needs into session_state
        # first, since the dropzone/settings widgets won't be re-rendered
        # once we switch into the scanning branch above.
        if run_btn and uploaded_files:
            st.session_state.pending_uploaded_files = uploaded_files
            st.session_state.pending_settings = {
                "ref_id": ref_id, "subject_name": subject_name, "incident_date": incident_date,
                "reported_weather": reported_weather, "lat": lat, "lon": lon, "owm_api_key": owm_api_key,
                "max_frames": max_frames, "ela_quality": ela_quality, "show_ela_image": show_ela_image,
            }
            st.session_state.is_scanning = True
            st.rerun()

else:
    # --- STATE 2: RESULTS DASHBOARD ---
    topbar_l2, topbar_c2, topbar_r2 = st.columns([1, 10, 1])
    with topbar_c2:
        st.markdown("""
            <div class="veda-topbar">
                <div class="veda-topbar-row">
                    <div class="logo-dot orb-3d-scene"><div class="orb-3d"><div class="orb-ring orb-ring-1"></div><div class="orb-ring orb-ring-2"></div><div class="orb-ring orb-ring-3"></div><div class="orb-core">✨</div></div></div>
                    <div class="logo-text">VeriLens AI</div>
                </div>
                <div class="logo-sub">Verification Report</div>
            </div>
        """, unsafe_allow_html=True)
    with topbar_r2:
        render_appearance_history_icons()

    if st.button("← New Verification", type="secondary"):
        st.session_state.is_analyzed = False
        st.rerun()

    st.markdown("""
    <div class="hero-card">
        <div class="glow-blob-2"></div>
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
            st.download_button(label="📥 Download PDF Report",
    data=pdf_bytes,
    file_name="deepfake_verification_report.pdf",
    mime="application/pdf",)
            st.divider()

        for r in results:
            # --- Step 1: big verdict hero — the one thing a viewer needs to see ---
            verdict_hero(r)

            col_media, col_info = st.columns([1, 1])
            with col_media:
                # Filename caption sits directly above its photo/video, no gap
                st.markdown(f'<p class="media-caption">📄 {r["file_name"]}</p>', unsafe_allow_html=True)
                if r["is_image"]: st.image(r["temp_path"], use_container_width=True)
                else: st.video(r["temp_path"])

            with col_info:
                # Dynamically retrieve labels for display
                current_labels = USE_CASES[r['use_case']]
                _details_rows = [
                    (current_labels["id_label"], r["ref_id"]),
                    (current_labels["name_label"], r["subject_name"] or "—"),
                    ("Use Case", r["use_case"]),
                    ("Analyzed", r["timestamp"]),
                ]
                _rows_html = "".join(
                    f'<div class="dc-row"><span class="dc-label">{label}</span><span class="dc-value">{value}</span></div>'
                    for label, value in _details_rows
                )
                st.markdown(f'<div class="details-card">{_rows_html}</div>', unsafe_allow_html=True)

            st.write("")

            with st.expander("🔍 View Full Forensic Breakdown", expanded=False):
                m1, m4, m5 = st.columns(3)
                _px = r.get("pixel_result") or {}
                _px_risk = "Anomalies Found" if any([_px.get("spectral_flag"), _px.get("too_smooth_flag"),
                                                       _px.get("patchy_flag"), _px.get("illum_jump_flag")]) else "Clean"
                metric_defs = [
                    (m1, "AI Classification", score_based_classification(r["score"])),
                    (m4, "Authenticity Score", f"{r['score']} / 100"),
                    (m5, "Pixel-Level Risk", _px_risk),
                ]
                _file_key = r["sha256"][:10]
                for _mi, (_col, _label, _value) in enumerate(metric_defs):
                    with _col:
                        with st.container(border=True, key=f"metric_card_{_file_key}_{_mi}"):
                            st.metric(_label, _value)

                # --- Pill-style tab row, built from plain st.button() instead of
                # st.tabs() — st.tabs()'s active-state styling silently never
                # applied (BaseWeb's internal structure isn't reliably
                # targetable via CSS across versions), so this reuses the
                # primary/secondary button styling that's already proven to
                # work everywhere else in this app. ---
                TAB_LABELS = ["🚨 Explainability", "🔬 File Forensics", "💥 Scene Plausibility",
                              "🧪 Advanced Forensics", "🌐 Context & Duplicate Check", "📤 Export Report"]
                _tab_key = f"active_tab_{_file_key}"
                if _tab_key not in st.session_state:
                    st.session_state[_tab_key] = 0

                tab_cols = st.columns(len(TAB_LABELS))
                for _ti, (_tcol, _tlabel) in enumerate(zip(tab_cols, TAB_LABELS)):
                    with _tcol:
                        if st.button(_tlabel, key=f"tabbtn_{_file_key}_{_ti}", use_container_width=True,
                                     type="primary" if st.session_state[_tab_key] == _ti else "secondary"):
                            st.session_state[_tab_key] = _ti
                            st.rerun()

                active_tab = st.session_state[_tab_key]
                st.write("")

                if active_tab == 0:
                    st.subheader("Identified Risk Indicators")
                    if r["flags"]:
                        for flag in r["flags"]: st.markdown(f'<div class="flag-card">{flag}</div>', unsafe_allow_html=True)
                        chart = render_deduction_chart(r["deductions"])
                        if chart: st.plotly_chart(chart, use_container_width=True)
                    else: st.markdown('<div class="clean-card">✅ No suspicious manipulation indicators detected.</div>', unsafe_allow_html=True)
                elif active_tab == 1:
                    ff1, ff2, ff3 = st.columns(3)
                    ff1.metric("File Type", "Image" if r["is_image"] else "Video")
                    ff2.metric("Encoder", r["metadata"]["encoder"] or "Unknown")
                    with ff3:
                        st.markdown('<p style="color:var(--ink-muted); font-size:0.82rem; font-weight:600; '
                                    'margin-bottom:0.3rem;">SHA-256</p>', unsafe_allow_html=True)
                        st.code(r["sha256"], language=None)
                    if r["is_image"] and r.get("ela_image") is not None:
                        st.image(r["ela_image"], use_container_width=True)
                        st.metric("ELA Score", f"{r['ela_score']:.2f}")
                    with st.expander("🔧 Developer Mode — raw metadata"):
                        st.json({"File Type": "Image" if r["is_image"] else "Video", "Encoder": r["metadata"]["encoder"], "SHA-256": r["sha256"]})
                elif active_tab == 2:
                    scene = r.get("scene_result")
                    if r["is_image"] or not scene:
                        st.info("Scene plausibility forensics (fire/smoke, plate, motion physics) apply to video only.")
                    else:
                        if scene["flags"]:
                            for flag in scene["flags"]:
                                st.markdown(f'<div class="flag-card">{flag}</div>', unsafe_allow_html=True)
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
                elif active_tab == 3:
                    px = r.get("pixel_result") or {}
                    px_flags = []
                    if px.get("spectral_flag"):
                        px_flags.append(f"🌀 Frequency-domain analysis found periodic upsampling artifacts "
                                         f"(anomaly score {px.get('spectral_anomaly_score', 0):.2f}, threshold 0.55) — "
                                         f"consistent with GAN/diffusion generation.")
                    if px.get("too_smooth_flag"):
                        px_flags.append(f"🧊 Noise floor is abnormally smooth (variance {px.get('noise_mean_var', 0):.2f}, "
                                         f"threshold 1.2) — consistent with AI generation or heavy denoising.")
                    if px.get("patchy_flag"):
                        px_flags.append(f"🧩 Noise residual is inconsistent across the frame "
                                         f"(coefficient of variation {px.get('noise_cv', 0):.2f}, threshold 1.6) — "
                                         f"possible splicing/compositing.")
                    if px.get("illum_jump_flag"):
                        px_flags.append(f"💡 Abrupt lighting jump detected ({px.get('max_illum_jump', 0):.1f} "
                                         f"luminance units) inconsistent with natural camera exposure.")

                    if px_flags:
                        for flag in px_flags:
                            st.markdown(f'<div class="flag-card">{flag}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="clean-card">✅ No frequency-domain, noise-residual, or '
                                     'illumination anomalies detected.</div>', unsafe_allow_html=True)

                    pf1, pf2, pf3 = st.columns(3)
                    pf1.metric("Spectral Anomaly Score", f"{px.get('spectral_anomaly_score', 0):.2f}")
                    pf2.metric("Noise Floor Variance", f"{px.get('noise_mean_var', 0):.2f}")
                    pf3.metric("Noise Patchiness (CV)", f"{px.get('noise_cv', 0):.2f}")

                    if px.get("has_content_credentials"):
                        st.success("🔏 Content Credentials (C2PA) metadata detected — this file carries "
                                    "cryptographic provenance/edit-history attestation. A corroborating positive "
                                    "signal, though not on its own proof of authenticity.")
                    else:
                        st.caption("No Content Credentials (C2PA) provenance metadata found. This is common and "
                                    "not itself suspicious — most media, real or fake, doesn't carry it yet.")

                    if not r["is_image"] and px.get("illum_series"):
                        st.write("**Frame-to-Frame Luminance**")
                        _lum_df = pd.DataFrame({"Frame #": list(range(len(px["illum_series"]))), "Mean Luminance": px["illum_series"]})
                        st.line_chart(_lum_df.set_index("Frame #"))

                    with st.expander("🔧 Developer Mode — raw pixel forensics data"):
                        st.json(px)
                elif active_tab == 4:
                    c1, c2 = st.columns(2)
                    with c1:
                        w = r["weather"]
                        match_icon = "✅" if w["matched"] else "⚠️"
                        st.markdown(f"**{match_icon} Reported weather:** {'Matches' if w['matched'] else 'Does not match'} conditions on record  \n"
                                    f"**Recorded conditions:** {w['api_weather']}  \n"
                                    f"**Note:** {w['note']}")
                        with st.expander("🔧 Developer Mode — raw weather data"):
                            st.json(w)
                    with c2: st.write("Duplicate detected" if r["duplicate_flag"] else "Clean from duplicates")
                else:
                    st.markdown(f"**Final Status:** {r['status']}")
                    st.markdown('<p style="color:var(--ink-muted); font-size:0.82rem; font-weight:600; '
                                'margin-bottom:0.3rem;">SHA-256</p>', unsafe_allow_html=True)
                    st.code(r["sha256"], language=None)

                    st.write("")
                    if _HAS_PDF_LIBS:
                        pdf_col1, pdf_col2 = st.columns([1, 1])
                        with pdf_col1:
                            pdf_bytes = generate_pdf_report(r)
                            _pdf_name = f"VeriLens_Report_{r['sha256'][:10]}.pdf"
                            st.download_button(
                                "📕 Download PDF Report", data=pdf_bytes, file_name=_pdf_name,
                                mime="application/pdf", use_container_width=True,
                                key=f"pdf_dl_{_file_key}",
                            )
                        with pdf_col2:
                            if st.button("👁️ Preview PDF Report", use_container_width=True, key=f"pdf_preview_btn_{_file_key}"):
                                st.session_state[f"show_pdf_preview_{_file_key}"] = True
                        if st.session_state.get(f"show_pdf_preview_{_file_key}"):
                            import base64 as _b64
                            _b64_pdf = _b64.b64encode(pdf_bytes).decode()
                            st.markdown(
                                f'<iframe src="data:application/pdf;base64,{_b64_pdf}" '
                                f'width="100%" height="650" style="border:1px solid var(--border); '
                                f'border-radius:12px;" type="application/pdf"></iframe>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("📕 PDF report unavailable — install `reportlab` and `matplotlib` "
                                   "(`pip install reportlab matplotlib`) to enable PDF exports.")

                    with st.expander("🔧 Developer Mode — raw export JSON"):
                        st.json({"status": r["status"], "score": r["score"], "sha256": r["sha256"]})
            st.divider()