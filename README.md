# 🕵️ Fake News Detection: An NLP & ML Study

A robust Natural Language Processing (NLP) and Machine Learning project designed to detect and classify fake news articles with high accuracy (over 94%). This project is optimized for portfolio showcases and addresses the critical problem of misinformation in the digital age.

---

## 🚀 Features

- **Advanced Text Preprocessing**: Automated cleaning including lowercase conversion, punctuation removal, tokenization, stemming (using Porter Stemmer), and stopword filtering.
- **High-Performance Classifier**: Leverages the **PassiveAggressiveClassifier**, a state-of-the-art algorithm for online-learning scenarios like social media feed analysis.
- **TF-IDF Vectorization**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) with bigram analysis to capture contextual patterns in fake news.
- **Interactive UI**: A sleek, user-friendly **Streamlit** dashboard for real-time fact-checking.
- **Explainable Results**: Features a detailed view of the cleaned tokens to see how the model "interprets" the input.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **NLP Library**: NLTK
- **Machine Learning**: Scikit-learn, Joblib
- **Data Manipulation**: Pandas, NumPy
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn

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
