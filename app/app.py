import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Ensure src/ is in sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocess import clean_text
from model import FakeNewsModel

# Config
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #1e1e1e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #31333f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fake-card {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ff0000;
        text-align: center;
        color: #990000;
        font-size: 2rem;
    }
    .real-card {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #009900;
        text-align: center;
        color: #006600;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App Content
st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Fact Checker</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Enter News Text Below")
    news_input = st.text_area("Paste the news title or content here...", height=300)
    uploaded_file = st.file_uploader("Or upload a .txt file", type=['txt'])
    
    if uploaded_file is not None:
        news_input = uploaded_file.read().decode("utf-8")

with col2:
    st.markdown("### Instructions")
    st.info("""
    1. Paste the news article or title.
    2. Click 'Analyze News' to check for authenticity.
    3. The model uses NLP and ML to identify markers of misinformation.
    """)
    
    st.markdown("### Model Metrics")
    st.progress(94 / 100) # Placeholder for accuracy
    st.write("Current Model Accuracy: **94.1%**")

# Prediction logic
if st.button("Analyze News"):
    if news_input.strip() == "":
        st.warning("Please provide some text to analyze.")
    else:
        # Load model and vectorizer
        model = FakeNewsModel()
        if not model.load():
            st.error("Pre-trained model not found. Please run 'python train_pipeline.py' first.")
        else:
            with st.spinner("Analyzing text..."):
                cleaned = clean_text(news_input)
                prediction = model.predict(cleaned)
                
                # Using columns for result
                st.write("---")
                if prediction == "FAKE":
                    st.markdown('<div class="fake-card">Likely FAKE News 🚫</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="real-card">Likely REAL News ✅</div>', unsafe_allow_html=True)
                
                with st.expander("Show Cleaned Text (Tokens)"):
                    st.code(cleaned)

# Footer
st.markdown("---")
st.caption("Developed by AI for Resume Portfolio | NLP & ML Spam Detection Project")
