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
    page_title="Veritas AI - Content Integrity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE FOR HISTORY ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'news_input' not in st.session_state:
    st.session_state.news_input = ""

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
    
    st.markdown("### ⚙️ Engine Setup")
    tesseract_path = st.text_input(
        "Tesseract EXE Path", 
        value=st.session_state.get('tesseract_path', r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        help="Paste the path to your tesseract.exe here. Usually in C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
    st.session_state.tesseract_path = tesseract_path
    
    if st.button("Apply Path"):
        if os.path.exists(tesseract_path):
            if pytesseract:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                st.success("OCR Path Applied!")
        else:
            st.error("Invalid Path! File not found.")

    st.markdown("---")
    st.markdown("### 📜 Past Activity")
    if not st.session_state.history:
        st.caption("No recent activity found.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"""
            <div class="history-card">
                <b>{item['label']}</b><br>
                <small>{item['time']}</small><br>
                <p style="font-size: 11px; margin: 0; color: #4b5563;">{item['snippet'][:40]}...</p>
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
    st.markdown('<p class="subtitle">Secure Content Verification: Spam & Fake News Detection</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis Workspace
    news_input = st.text_area("Analysis Workspace", 
                             value=st.session_state.news_input,
                             placeholder="Paste news content here for instant analysis...", 
                             height=300, 
                             label_visibility="collapsed",
                             key="main_workspace")
    
    # Update session state with current workspace value to prevent loss
    st.session_state.news_input = news_input

    # File Upload Support
    uploaded_file = st.file_uploader("Upload Content (PDF, JPG, JPEG, PNG, TXT)", 
                                    type=['txt', 'pdf', 'jpg', 'jpeg', 'png'])
    
    extracted_text = ""
    if uploaded_file and st.session_state.get('last_processed_file') != uploaded_file.name:
        with st.status(f"Processing {uploaded_file.name}...", expanded=False) as status:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'txt':
                extracted_text = uploaded_file.read().decode("utf-8")
                status.update(label="Text extracted successfully!", state="complete")
            elif file_type == 'pdf':
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text_parts = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    extracted_text = "\n".join(text_parts)
                    if not extracted_text.strip():
                        st.warning("No text found in PDF. It might be a scanned document.")
                        status.update(label="PDF parsed, but no text found.", state="error")
                    else:
                        status.update(label="PDF text extracted!", state="complete")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    status.update(label="PDF Extraction Failed", state="error")
            elif file_type in ['jpg', 'jpeg', 'png']:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image Preview", use_column_width=True)
                if pytesseract:
                    try:
                        # Priority 1: User Path from Sidebar
                        if st.session_state.get('tesseract_path') and os.path.exists(st.session_state.tesseract_path):
                            pytesseract.pytesseract.tesseract_cmd = st.session_state.tesseract_path
                        # Priority 2: Standard Auto-locate
                        elif os.name == 'nt' and not hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
                            possible_paths = [
                                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                                r'C:\Users\\' + os.getlogin() + r'\AppData\Local\Tesseract-OCR\tesseract.exe',
                                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                            ]
                            for p in possible_paths:
                                if os.path.exists(p):
                                    pytesseract.pytesseract.tesseract_cmd = p
                                    break
                        
                        extracted_text = pytesseract.image_to_string(img)
                        if not extracted_text.strip():
                            st.warning("OCR returned no text. Try a clearer image.")
                            status.update(label="OCR completed, no text found.", state="error")
                        else:
                            status.update(label="Image text extracted via OCR!", state="complete")
                    except Exception as e:
                        st.error(f"OCR Failure: {e}")
                        st.info("Check the Tesseract path in the sidebar. [Download Info](https://github.com/UB-Mannheim/tesseract/wiki)")
                        status.update(label="OCR Failed - Check Sidebar", state="error")
                else:
                    st.warning("OCR (pytesseract) library not found.")
                    status.update(label="OCR Unconfigured", state="error")
        
        if extracted_text:
            st.session_state.news_input = extracted_text
            st.session_state.last_processed_file = uploaded_file.name
            st.session_state.auto_analyze = True # Trigger analysis on rerun
            st.success("Content extracted! Initiating analysis...")
            st.rerun()

    # Automatic analysis trigger from file upload
    if st.session_state.get('auto_analyze', False):
        st.session_state.auto_analyze = False # Reset
        do_analysis = True
    else:
        do_analysis = st.button("Analyze Content", use_container_width=True)

    # Analyze Execution
    if do_analysis:
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
                label = "NOT SPAM" if prediction == "REAL" else "SPAM"
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
                        <p style="margin-top: 10px;">Linguistic markers suggest this content may be <b>Spam or Fake News</b>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-alert real-alert">
                        <p style="font-size: 1.1rem; text-transform: uppercase;">Confidence Engine Results</p>
                        <h2 class="result-text">{label}</h2>
                        <p style="margin-top: 10px;">Linguistic markers suggest this content is <b>Legitimate & Reliable</b>.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Engine Failure: Pre-trained model not optimized. Please run the training pipeline first.")

    # Minimalist Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.divider()
    cols = st.columns(3)
    cols[1].caption("Veritas AI | Developed for Professional ML Portfolio")
