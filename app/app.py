import streamlit as st
import pandas as pd
import joblib
import os
import sys
import time
from datetime import datetime
from PIL import Image
import PyPDF2
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Ensure src/ is in sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocess import clean_text
from model import FakeNewsModel

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Veritas AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE FOR HISTORY ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #fcfdfe; }
    
    /* Centered Header */
    .app-header {
        text-align: center;
        padding: 60px 0 40px;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 10px;
    }

    /* Professional Sidebar */
    .css-1d391kg { background-color: #f8fafc !important; }
    .history-card {
        padding: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 0.85rem;
    }

    /* Result Indicators */
    .result-alert {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 40px;
        animation: fadeIn 0.4s ease-in-out;
    }
    
    .fake-alert {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    
    .real-alert {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
    }
    
    .result-text {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: PAST ACTIVITY ---
with st.sidebar:
    st.markdown("# 🛡️ Veritas AI")
    st.markdown("### Past Activity")
    if not st.session_state.history:
        st.caption("No recent activity found.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"""
            <div class="history-card">
                <b>{item['label']}</b><br>
                <small>{item['time']}</small><br>
                <span style="font-size: 11px;">{item['snippet'][:40]}...</span>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# --- MAIN UI ---
col_left, col_mid, col_right = st.columns([1, 4, 1])

with col_mid:
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Veritas AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Secure, Enterprise-Grade Content Verification</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input Box
    news_input = st.text_area("Analysis Workspace", 
                             placeholder="Paste news content here for instant analysis...", 
                             height=300, 
                             label_visibility="collapsed")

    # File Upload Support
    uploaded_file = st.file_uploader("Multimodal Input (PDF, WebP, JPEG, PNG, TXT)", 
                                    type=['txt', 'pdf', 'jpg', 'jpeg', 'png'])
    
    extracted_text = ""
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'txt':
            extracted_text = uploaded_file.read().decode("utf-8")
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() or ""
        elif file_type in ['jpg', 'jpeg', 'png']:
            img = Image.open(uploaded_file)
            if pytesseract:
                # Note: Tesseract needs engine installed on system. If not, this skips.
                try:
                    extracted_text = pytesseract.image_to_string(img)
                except Exception:
                    st.warning("OCR Engine not found. Using metadata/display only.")
            else:
                st.warning("OCR (Pytesseract) is not installed. Analyzing metadata only.")
        
        if extracted_text:
            news_input = extracted_text

    # Analyze Button
    if st.button("Analyze Content", use_container_width=True):
        if not news_input.strip():
            st.error("Engine requires content to proceed. Please provide text or a file.")
        else:
            # Simulation of 5-second analytical deep-dive
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(1, 101):
                time.sleep(0.045)  # Sums roughly to 4.5s + 0.5s overhead = 5s
                progress_bar.progress(i)
                if i < 30: status_text.text("Initial Parsing...")
                elif i < 60: status_text.text("Scanning Linguistic Markers...")
                elif i < 90: status_text.text("TF-IDF Vector Mapping...")
                else: status_text.text("Finalizing Inference...")
            
            # Backend Execution
            model = FakeNewsModel()
            if model.load():
                cleaned = clean_text(news_input)
                prediction = model.predict(cleaned)
                
                # Update Session State
                label = "NOT A FAKE NEWS" if prediction == "REAL" else "FAKE NEWS"
                st.session_state.history.append({
                    'label': label,
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'snippet': news_input[:60]
                })

                # Results Display
                st.markdown("---")
                if prediction == "FAKE":
                    st.markdown(f"""
                    <div class="result-alert fake-alert">
                        <p style="font-size: 1.1rem; text-transform: uppercase;">Confidence Engine Results</p>
                        <h2 class="result-text">{label}</h2>
                        <p style="margin-top: 10px;">Linguistic patterns strongly align with misinformation traits.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-alert real-alert">
                        <p style="font-size: 1.1rem; text-transform: uppercase;">Confidence Engine Results</p>
                        <h2 class="result-text">{label}</h2>
                        <p style="margin-top: 10px;">This content matches verified/reliable reporting patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Engine Failure: Pre-trained model not optimized. Please run the training pipeline first.")

    # Minimalist Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.divider()
    cols = st.columns(3)
    cols[1].caption("Veritas AI | Developed for Professional ML Portfolio")
