# 🕵️ Fake News Detection: An NLP & ML Study

A robust Natural Language Processing (NLP) and Machine Learning project designed to detect and classify fake news articles with high accuracy (over 94%). This project is optimized for portfolio showcases and addresses the critical problem of misinformation in the digital age.

---

## 🚀 Features

- **Advanced Text Preprocessing**: Automated cleaning including lowercase conversion, punctuation removal, tokenization, stemming (using Porter Stemmer), and stopword filtering.
- **High-Performance Classifier**: Leverages the **PassiveAggressiveClassifier**, a state-of-the-art algorithm for online-learning scenarios like social media feed analysis.
- **TF-IDF Vectorization**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) with bigram analysis to capture contextual patterns in fake news.
- **Multimodal Support**: Support for Text, PDF, JPG, JPEG, and PNG files with automated content extraction.
- **Spam & Fake News Classification**: Real-time classification into **SPAM** or **NOT SPAM** categories.
- **Explainable Results**: Features a detailed view of results with a professional animation and progress tracking.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **NLP Library**: NLTK
- **Machine Learning**: Scikit-learn, Joblib
- **Data Manipulation**: Pandas, NumPy
- **Frontend**: Streamlit
- **Multimodal**: PyPDF2 (PDF), Pytesseract (OCR for Images)
- **Visualization**: Matplotlib, Seaborn

---

## 🖼️ Multimodal Input Setup (OCR)

To process images (JPG/JPEG/PNG), you need to install the Tesseract OCR engine on your system:

### 1. Install Tesseract Engine
- **Windows**: Download the installer from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 2. Configure in App
1. Open the **Veritas AI** application.
2. Expand the **Sidebar** (`>` arrow at top left).
3. Under **⚙️ Engine Setup**, enter the path to your `tesseract.exe`.
   - Default Windows Path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. Click **Apply Path**.

Now, when you upload any image or PDF, Veritas AI will automatically extract the text and classify it as **SPAM** or **NOT SPAM**.

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.66% |
| **Precision** | 98.3% |
| **Recall** | 98.9% |
| **F1-Score** | 98.6% |

---

## 📦 Project Structure

```bash
├── app/
│   └── app.py              # Streamlit Web Application
├── data/
│   └── download_data.py    # Dataset acquisition script
├── models/                 # Saved models and vectorizers
├── src/
│   ├── preprocess.py       # NLP Cleaning pipeline
│   └── model.py            # Model architecture and training logic
├── train_pipeline.py       # End-to-end training orchestrator
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 🏃 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data & Train Model
```bash
python data/download_data.py
python train_pipeline.py
```

### 3. Launch App
```bash
streamlit run app/app.py
```

---

## 📝 License
This project is open-source and intended for educational and portfolio purposes.
