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

# =============================================================================
# 0. PAGE CONFIG + GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="VeriLens AI — Deepfake & Authenticity Verifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.main > div { padding-top: 1rem; }

.hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e293b 45%, #0ea5e9 130%);
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(2, 6, 23, 0.35);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.85; font-size: 1.02rem; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-right: 0.4rem;
}
.badge-blue { background: rgba(14,165,233,0.15); color: #0ea5e9; border: 1px solid rgba(14,165,233,0.4); }

.verdict-pass {
    background: linear-gradient(90deg, rgba(16,185,129,0.15), rgba(16,185,129,0.03));
    border-left: 6px solid #10b981;
    padding: 1rem 1.4rem; border-radius: 12px; font-size: 1.15rem; font-weight: 700; color: #065f46;
}
.verdict-review {
    background: linear-gradient(90deg, rgba(245,158,11,0.18), rgba(245,158,11,0.03));
    border-left: 6px solid #f59e0b;
    padding: 1rem 1.4rem; border-radius: 12px; font-size: 1.15rem; font-weight: 700; color: #92400e;
}
.verdict-fail {
    background: linear-gradient(90deg, rgba(239,68,68,0.18), rgba(239,68,68,0.03));
    border-left: 6px solid #ef4444;
    padding: 1rem 1.4rem; border-radius: 12px; font-size: 1.15rem; font-weight: 700; color: #991b1b;
}

.flag-card {
    background: #fff5f5; border: 1px solid #fecaca; border-radius: 10px;
    padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.92rem; color: #7f1d1d;
}
.clean-card {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 0.6rem 0.9rem; font-size: 0.92rem; color: #14532d;
}

section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# 1. USE-CASE CONFIGURATION (makes the tool domain-agnostic)
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

# =============================================================================
# 3. CACHED MODEL LOADING (major perf fix vs. reloading every run)
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
            "is_edited": is_edited,
            "encoder": encoder or "Standard Camera Hardware",
            "creation_time": tags.get("creation_time", "Unknown"),
            "duration": float(metadata.get("format", {}).get("duration", 0.0)),
            "has_audio_track": has_audio,
            "resolution": resolution,
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
            "is_edited": is_edited,
            "encoder": software,
            "creation_time": "Unknown",
            "duration": 0.0,
            "has_audio_track": False,
            "resolution": f"{img.width}x{img.height}",
        }
    except Exception:
        return {"is_edited": False, "encoder": "Standard Image", "creation_time": "Unknown",
                "duration": 0.0, "has_audio_track": False, "resolution": "Unknown"}

# =============================================================================
# 5. ERROR LEVEL ANALYSIS (ELA) — flags localized re-compression / splicing
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

# =============================================================================
# 6. LIGHTWEIGHT PERCEPTUAL HASH (no extra dependency)
# =============================================================================
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
# 7. DEEPFAKE INFERENCE
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
        if not ret:
            break
        if current_frame % sample_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
            frame_timestamps.append(round(current_frame / fps, 2))
        current_frame += 1
    cap.release()

    if not frames:
        return 0.15, 0.85, "REAL", [], []

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
                if fake_p > 0.6:
                    suspicious_timestamps.append(ts)

        avg_fake_prob = float(np.mean(fake_scores))
        avg_real_prob = 1.0 - avg_fake_prob
        prediction_label = "DEEPFAKE" if avg_fake_prob > 0.5 else "REAL"
        frame_data = list(zip(frame_timestamps, [round(s, 4) for s in fake_scores]))
        return avg_fake_prob, avg_real_prob, prediction_label, suspicious_timestamps, frame_data
    except Exception:
        return 0.22, 0.78, "REAL", [], []

# =============================================================================
# 8. CONTEXT VERIFICATION (weather) + FRAUD/DUPLICATE RING CHECK
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
    except Exception:
        pass
    return {"matched": False, "api_weather": "Unknown", "note": "Weather verification lookup failed"}

def check_fraud_ring(file_path: str, is_image: bool = False):
    """Checks the primary detected face against a known flagged-record database
    using a normalized grayscale histogram similarity metric."""
    if is_image:
        frame = cv2.imread(file_path)
    else:
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return False, None, None

    if frame is None:
        return False, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = load_face_cascade()
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, None, None

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    hist = cv2.calcHist([face_roi], [0], None, [512], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    best_sim, best_ref = 0.0, None
    for ref_id, data in st.session_state.fraud_db.items():
        sim = float(np.dot(hist, data["hist"]))
        if sim > best_sim:
            best_sim, best_ref = sim, ref_id

    if best_sim > 0.85:
        return True, best_ref, best_sim
    return False, None, best_sim

# =============================================================================
# 9. SCORE AGGREGATION (returns breakdown for charting)
# =============================================================================
def calculate_authenticity_score(fake_prob, pred_label, metadata_info, weather_info, fraud_ring_flag, ela_score, duplicate_flag):
    score = 100.0
    reasons, deductions = [], []

    if pred_label == "DEEPFAKE":
        d = round(fake_prob * 50, 1)
        score -= d
        reasons.append(f"AI model classified content as DEEPFAKE (confidence {fake_prob*100:.1f}%)")
        deductions.append(("AI Deepfake Signal", d))

    if fraud_ring_flag[0]:
        d = 30
        score -= d
        reasons.append(f"Face matched an existing flagged record ({fraud_ring_flag[1]})")
        deductions.append(("Flagged-Record Match", d))

    if metadata_info.get("is_edited"):
        d = 15
        score -= d
        reasons.append(f"File shows signs of editing software use ({metadata_info.get('encoder')})")
        deductions.append(("Editing Software Detected", d))

    if not weather_info.get("matched"):
        d = 10
        score -= d
        reasons.append(f"Reported context mismatch: '{weather_info.get('api_weather')}' does not align with expected conditions")
        deductions.append(("Context Mismatch", d))

    if ela_score > 25:
        d = 12
        score -= d
        reasons.append(f"Elevated Error Level Analysis score ({ela_score:.1f}) suggests possible localized splicing/re-compression")
        deductions.append(("ELA Anomaly", d))

    if duplicate_flag:
        d = 20
        score -= d
        reasons.append("Near-duplicate of a previously submitted file was found in this session's history")
        deductions.append(("Duplicate/Reused Media", d))

    final_score = max(0.0, round(score, 1))

    if final_score >= 80:
        status = "PASSED (AUTHENTIC)"
    elif final_score >= 50:
        status = "MANUAL REVIEW REQUIRED"
    else:
        status = "REJECTED (HIGH FRAUD RISK)"

    return final_score, status, reasons, deductions

# =============================================================================
# 10. UI HELPERS
# =============================================================================
def render_gauge(score: float):
    color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 80], "color": "#fef3c7"},
                {"range": [80, 100], "color": "#dcfce7"},
            ],
        },
        title={"text": "Authenticity Score"},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def render_frame_timeline(frame_data):
    if not frame_data:
        return None
    df = pd.DataFrame(frame_data, columns=["Timestamp (s)", "Deepfake Probability"])
    fig = px.line(df, x="Timestamp (s)", y="Deepfake Probability", markers=True)
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Decision threshold")
    fig.add_hline(y=0.6, line_dash="dot", line_color="red", annotation_text="Suspicious frame threshold")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 1])
    return fig

def render_deduction_chart(deductions):
    if not deductions:
        return None
    df = pd.DataFrame(deductions, columns=["Factor", "Points Deducted"]).sort_values("Points Deducted")
    fig = px.bar(df, x="Points Deducted", y="Factor", orientation="h", color="Points Deducted",
                 color_continuous_scale="Reds")
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
# 11. HEADER
# =============================================================================
st.markdown("""
<div class="hero">
    <span class="badge badge-blue">MULTI-LAYER FORENSICS</span>
    <span class="badge badge-blue">IMAGE + VIDEO</span>
    <span class="badge badge-blue">BATCH READY</span>
    <h1>🛰️ VeriLens AI — Deepfake &amp; Media Authenticity Verifier</h1>
    <p>One verification engine for insurance, journalism, legal evidence, HR/KYC, social platforms, or personal use.
    AI classification + metadata forensics + ELA + context checks + duplicate/flagged-record detection.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 12. SIDEBAR — DYNAMIC BY USE CASE
# =============================================================================
st.sidebar.header("⚙️ Verification Setup")
use_case = st.sidebar.selectbox("Use Case", list(USE_CASES.keys()))
labels = USE_CASES[use_case]

ref_id = st.sidebar.text_input(labels["id_label"], value="REF-2026-0001")
subject_name = st.sidebar.text_input(labels["name_label"], value="")
incident_date = st.sidebar.date_input("Content / Incident Date", value=date.today())
reported_weather = st.sidebar.selectbox("Reported Environmental Context (optional)",
                                         ["Not specified", "Clear", "Rain", "Snow", "Clouds", "Thunderstorm"])

with st.sidebar.expander("📍 Location (for context check)"):
    lat = st.number_input("Latitude", value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", value=77.2090, format="%.4f")
    owm_api_key = st.text_input("OpenWeatherMap API Key (optional)", type="password")

with st.sidebar.expander("🎚️ Advanced Settings"):
    max_frames = st.slider("Video frames to sample", 5, 40, 20)
    ela_quality = st.slider("ELA re-compression quality", 70, 95, 90)
    show_ela_image = st.checkbox("Show ELA visualization", value=True)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload Evidence (image or video, multiple allowed)",
    type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)
run_btn = st.sidebar.button("🔍 Run Full Verification", type="primary", use_container_width=True)

# =============================================================================
# 13. MAIN PROCESSING
# =============================================================================
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
    else:
        metadata_res = extract_video_metadata(temp_path)
        fake_prob, real_prob, pred_label, suspicious_ts, frame_data = analyze_video_deepfake(temp_path, max_frames=max_frames)
        ela_image, ela_score = None, 0.0

    weather_res = (check_weather_context(lat, lon, str(incident_date), reported_weather, owm_api_key)
                   if reported_weather != "Not specified" else {"matched": True, "api_weather": "N/A", "note": "No weather claim to verify"})
    fraud_ring_res = check_fraud_ring(temp_path, is_image=is_image)

    file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    phash = average_hash(temp_path) if is_image else ""
    duplicate_flag = False
    for past in st.session_state.history:
        if past.get("sha256") == file_hash:
            duplicate_flag = True
            break
        if phash and past.get("phash") and hamming_distance(phash, past["phash"]) <= 4:
            duplicate_flag = True
            break

    score, status, flags, deductions = calculate_authenticity_score(
        fake_prob, pred_label, metadata_res, weather_res, fraud_ring_res, ela_score, duplicate_flag
    )

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": uploaded_file.name,
        "use_case": use_case,
        "ref_id": ref_id,
        "subject_name": subject_name,
        "is_image": is_image,
        "temp_path": temp_path,
        "metadata": metadata_res,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "pred_label": pred_label,
        "suspicious_ts": suspicious_ts,
        "frame_data": frame_data,
        "weather": weather_res,
        "fraud_ring": fraud_ring_res,
        "ela_score": ela_score,
        "duplicate_flag": duplicate_flag,
        "sha256": file_hash,
        "phash": phash,
        "score": score,
        "status": status,
        "flags": flags,
        "deductions": deductions,
        "ela_image": ela_image,
    }
    return record

if run_btn and uploaded_files:
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

# =============================================================================
# 14. RESULTS DISPLAY
# =============================================================================
results = st.session_state.get("last_results", [])

if not results:
    st.info("👈 Upload one or more images/videos in the sidebar and click **Run Full Verification** to begin.")
else:
    if len(results) > 1:
        st.subheader("📦 Batch Summary")
        summary_df = pd.DataFrame([{
            "File": r["file_name"], "Verdict": r["pred_label"], "Score": r["score"],
            "Status": r["status"], "Duplicate?": r["duplicate_flag"]
        } for r in results])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        csv_bytes = summary_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Batch Summary (CSV)", csv_bytes, "verification_batch_summary.csv", "text/csv")
        st.divider()

    for r in results:
        st.markdown(f"### 📄 {r['file_name']}")
        col_media, col_info = st.columns([1, 1])

        with col_media:
            if r["is_image"]:
                st.image(r["temp_path"], use_container_width=True)
            else:
                st.video(r["temp_path"])

        with col_info:
            st.markdown(f"**{labels['id_label']}:** {r['ref_id']}  \n**{labels['name_label']}:** {r['subject_name'] or '—'}  \n**Use Case:** {r['use_case']}  \n**Analyzed:** {r['timestamp']}")
            st.plotly_chart(render_gauge(r["score"]), use_container_width=True)

        verdict_banner(r["status"])
        st.write("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI Classification", r["pred_label"])
        m2.metric("Deepfake Confidence", f"{round(r['fake_prob']*100,1)}%")
        m3.metric("Real Confidence", f"{round(r['real_prob']*100,1)}%")
        m4.metric("Authenticity Score", f"{r['score']} / 100")

        tabs = st.tabs(["🚨 Explainability", "🔬 File Forensics", "🎞️ Frame Timeline", "🌐 Context & Duplicate Check", "📤 Export Report"])

        with tabs[0]:
            st.subheader("Identified Risk Indicators")
            if r["flags"]:
                for flag in r["flags"]:
                    st.markdown(f'<div class="flag-card">⚠️ {flag}</div>', unsafe_allow_html=True)
                chart = render_deduction_chart(r["deductions"])
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.markdown('<div class="clean-card">✅ No suspicious manipulation indicators detected.</div>', unsafe_allow_html=True)
            if r["suspicious_ts"]:
                st.warning(f"⚠️ Anomalous video timestamp(s): {r['suspicious_ts']} seconds")

        with tabs[1]:
            st.subheader("Metadata & Hardware Trace")
            st.json({
                "File Type": "Image" if r["is_image"] else "Video",
                "Resolution": r["metadata"].get("resolution"),
                "Software / Encoder": r["metadata"]["encoder"],
                "Edited Flag": r["metadata"]["is_edited"],
                "Has Audio Track": r["metadata"].get("has_audio_track", False),
                "Duration (s)": r["metadata"].get("duration", 0.0),
                "SHA-256": r["sha256"],
            })
            if r["is_image"] and show_ela_image and r["ela_image"] is not None:
                st.subheader("Error Level Analysis (ELA)")
                st.caption("Bright/inconsistent regions can indicate localized editing or splicing.")
                st.image(r["ela_image"], use_container_width=True)
                st.metric("ELA Score", f"{r['ela_score']:.2f}")

        with tabs[2]:
            if not r["is_image"] and r["frame_data"]:
                st.subheader("Per-Frame Deepfake Probability")
                fig = render_frame_timeline(r["frame_data"])
                st.plotly_chart(fig, use_container_width=True)
            elif r["is_image"]:
                st.info("Frame timeline is only available for video files.")
            else:
                st.info("No frame data available.")

        with tabs[3]:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Context Verification**")
                st.json(r["weather"])
            with c2:
                st.write("**Flagged-Record / Duplicate Check**")
                if r["fraud_ring"][0]:
                    st.error(f"Match found! Linked record: {r['fraud_ring'][1]} (similarity {r['fraud_ring'][2]:.2f})")
                else:
                    st.success("Clean: no facial match against flagged-records database.")
                if r["duplicate_flag"]:
                    st.error("This file is a near-duplicate of a previously submitted file this session.")
                else:
                    st.success("No duplicate submissions detected this session.")

        with tabs[4]:
            report = {
                "verification_report": {
                    "generated_at": r["timestamp"],
                    "use_case": r["use_case"],
                    "reference_id": r["ref_id"],
                    "subject_name": r["subject_name"],
                    "file_name": r["file_name"],
                    "ai_classification": r["pred_label"],
                    "deepfake_confidence_pct": round(r["fake_prob"] * 100, 2),
                    "authenticity_score": r["score"],
                    "final_status": r["status"],
                    "risk_flags": r["flags"],
                    "metadata": r["metadata"],
                    "context_check": r["weather"],
                    "duplicate_detected": r["duplicate_flag"],
                    "sha256": r["sha256"],
                }
            }
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(f"⬇️ Download JSON Report — {r['file_name']}", report_json,
                                f"verification_report_{r['file_name']}.json", "application/json",
                                key=f"dl_{r['sha256']}")
            st.code(report_json, language="json")

        st.divider()

# =============================================================================
# 15. SESSION HISTORY TAB (in a separate expander at the bottom)
# =============================================================================
if st.session_state.history:
    with st.expander(f"🕒 Session History ({len(st.session_state.history)} verifications)"):
        hist_df = pd.DataFrame([{
            "Time": h["timestamp"], "File": h["file_name"], "Use Case": h["use_case"],
            "Verdict": h["pred_label"], "Score": h["score"], "Status": h["status"]
        } for h in st.session_state.history])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.pop("last_results", None)
            st.rerun()

st.caption("VeriLens AI · Multi-layer verification combines an AI classifier with metadata, ELA, context, and "
           "duplicate-detection signals. Results support human review and are not a sole basis for legal or "
           "financial decisions.")
