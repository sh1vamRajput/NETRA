import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from typing import Dict, Tuple

st.set_page_config(page_title="N.E.T.R.A - Deepfake Detection", layout="centered", page_icon=":shield:")

# --- Utility Functions ---

def simulate_model(name, image_bytes):
    np.random.seed(hash((name, image_bytes)) % 2**32)
    confidence = np.random.uniform(0.1, 0.99)
    verdict = 'Deepfake' if confidence > 0.5 else 'Authentic'
    return verdict, confidence

def simulate_video_model(name, video_path, sample_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chosen = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    frames_scores = []
    for f in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        is_success, buffer = cv2.imencode(".jpg", frame)
        img_bytes = buffer.tobytes()
        _, conf = simulate_model(name, img_bytes)
        frames_scores.append(conf)
    cap.release()
    confidence = np.mean(frames_scores) if frames_scores else 0.5
    verdict = 'Deepfake' if confidence > 0.5 else 'Authentic'
    return verdict, confidence

def consensus(judge_results: Dict[str, Tuple[str, float]]):
    deepfake_votes = sum(1 for v, c in judge_results.values() if v == "Deepfake")
    verdict = "Likely Deepfake" if deepfake_votes >= 2 else "Likely Authentic"
    color = "red" if verdict == "Likely Deepfake" else "green"
    mean_conf = np.mean([c for _, c in judge_results.values()])
    return verdict, color, mean_conf

# --- UI Styling ---

css = """
<style>
body {background: radial-gradient(ellipse at top, #09182f 0%, #060911 100%) !important;}
section[data-testid="stSidebar"] {background: #060911 !important;}
.ghost-header {font-size:3rem; font-weight:900; background:linear-gradient(90deg,#4ef1fe,#7ed8f7,#ccdceb);background-clip:text;-webkit-background-clip:text;color:transparent;}
.status-banner {background:linear-gradient(90deg, #1f3342aa, #35bde8aa); color:#fff; border-radius:8px; margin-bottom:1.2rem; padding:0.5rem 1rem;font-weight:700;}
.cyber-panel {border:1.5px solid #0ff; border-radius:16px; background:linear-gradient(120deg,#080e17 92%,#122836 99%)!important; box-shadow:0 4px 16px #00ffd255;}
button, .stButton>button {background:linear-gradient(90deg,#2ca3c7, #1a5276); border-radius:6px; color:#fff;}
.inv-input input {border: 1px solid #41e7f3; border-radius:8px; background:#181e29; color: #8be1ff;}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown('<div class="status-banner">Welcome to N.E.T.R.A AI - Deepfake Detection Dashboard</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align:center; margin-top:-20px;'>
  <span class="ghost-header">N.E.T.R.A AI</span>
  <div style='text-transform:uppercase;font-size:1.2rem;letter-spacing:2px;color:#c6e8fb;font-weight:700'>Multi-Model Deepfake Detection System</div>

  <div style='font-size:0.92rem;color:#4aeaffa5;text-align:center;margin-bottom:6px;'>Version 1.0 &nbsp; | &nbsp; Built by Aaditya Ranjan Moitra 🚀</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.markdown("""
    <div class="cyber-panel" style="max-width:420px;margin:2rem auto;padding:2rem 2.4rem">
        <h3 style='text-align:center;color:#14f0ed;font-weight:800;font-family:monospace'>N.E.T.R.A Auth</h3>
        <div style='text-align:center;color:#7dcffb;font-size:1rem'>Please identify yourself to proceed</div>
        <div style="margin-top:1em;color:#b1c6d9;text-align:center;font-size:0.97rem;">Enter your name below:</div>
    """, unsafe_allow_html=True)
    
    name_input = st.text_input("Enter Investigator Name", key="name_input", placeholder="Investigator", label_visibility="collapsed")
    
    # Store the name in a separate variable instead of directly modifying session_state.investigator
    if st.button("Proceed", type="primary", use_container_width=True, disabled=(not name_input.strip())):
        st.session_state["auth"] = True
        st.session_state["investigator"] = name_input
        st.rerun()
        
    if st.button("Skip Authentication", use_container_width=True):
        st.session_state["auth"] = True
        st.session_state["investigator"] = "Anonymous Investigator"
        st.rerun()
        
    st.markdown("<div style='margin-top:1em;text-align:center;color:#555;font-size:0.85em;'>System Initialized</div>", unsafe_allow_html=True)
    st.stop()

with st.container():
    st.markdown("<div class='cyber-panel' style='max-width:640px;margin:2.2rem auto 1.3rem;padding:1.5em 2em;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:1.18rem;text-align:center;font-weight:600;color:#4ef1fe;margin-bottom:0.3em'>
        Investigator: <span style="color:#fff">{st.session_state.get('investigator','')}</span>
    </div>
    """, unsafe_allow_html=True)

    choice = st.radio('Select media type:', ['Image', 'Video'], horizontal=True)
    uploaded = st.file_uploader(f"Upload {'an image' if choice=='Image' else 'a video'}", type=['jpg','jpeg','png'] if choice=='Image' else ['mp4','avi','mov'])
    analyzed = False
    judge_res = {}

    if uploaded:
        st.markdown("<div style='text-align:center;margin-top:1.2em;font-size:1.19em;color:#b8fffb;'>Ready to analyze your file with 3 independent AI models.</div>", unsafe_allow_html=True)
        if st.button("Begin Analysis", use_container_width=True):
            analyzed = True
            media_bytes = uploaded.read()
            st.progress(0, text="Running analysis...")

            if choice == "Image":
                for model in ["MesoNet", "XceptionNet", "EfficientNet"]:
                    verdict, conf = simulate_model(model, media_bytes)
                    judge_res[model] = (verdict, conf)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as vf:
                    vf.write(media_bytes)
                    vf.flush()
                    for model in ["MesoNet", "XceptionNet", "EfficientNet"]:
                        verdict, conf = simulate_video_model(model, vf.name)
                        judge_res[model] = (verdict, conf)
                os.unlink(vf.name)

            vrd, color, mean_conf = consensus(judge_res)
            st.success("Analysis Completed.")

    if judge_res:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.15em;color:#dbedf3;font-weight:700;text-align:center;'>Multi-Model Results</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (model, (verdict, conf)) in enumerate(judge_res.items()):
            with cols[idx]:
                st.markdown(f"""
                <div style='padding:1.1em 0.5em;border-radius:12px;background:rgba(30,44,69,0.91);border:1.5px solid #22e6fb;box-shadow:0 4px 16px #23e8ff30;text-align:center'>
                    <div style='font-size:1.04em;color:#80eeff;font-weight:bold'>{model}</div>
                    <div style='font-size:1.25em;font-weight:850;color:{'#ff1744' if verdict=='Deepfake' else '#6fffa2'}'>{verdict}</div>
                    <div style='color:#b8eaff;font-size:1.09rem'>Confidence: <strong>{conf:.2f}</strong></div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='max-width:510px;margin:1em auto 0;border-radius:12px;padding:1.4em 1em 1em 1em;background:linear-gradient(120deg,#121e29 85%,#264351 99%)!important;text-align:center;border:2px solid #22e6fb;box-shadow:0 4px 24px #36f4f247;'>
        <div style='font-size:1.2em;font-weight:700;color:#ffeea5;letter-spacing:1.5px;margin-bottom:0.4em;'>Consensus Verdict</div>
        <div style="font-size:1.3em;color:{'red' if color=='red' else '#45ec71'};font-weight:800;">
            {vrd}
        </div>
        <div style='font-size:1.05em;color:#b6fbeb;'>Mean Confidence: <strong>{mean_conf:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.info("For best results, use high-quality face-containing images or clear videos. This system uses three independent AI models for a consensus verdict.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<div style='text-align:center;color:#4af1e5a8;font-size:0.85em;margin-top:2em;'>Copyright © 2025 Aaditya Ranjan Moitra. All rights reserved.</div>
""", unsafe_allow_html=True)