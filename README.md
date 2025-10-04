# 🎭 Emotion & Sentiment Detection with Streamlit

A web application built using **Streamlit** to detect emotions and sentiment from Indonesian text.  
This project uses **Machine Learning models** (Naive Bayes, Random Forest, and SVM) trained with **TF-IDF vectorization** and stored as `.pkl` files for direct use.

---

## 📂 Project Structure

project-root/
│
├── app.py # Main Streamlit app
│
├── encoder/ # Label encoders
│ ├── label_encoder_emotion.pkl
│ └── label_encoder_sentiment.pkl
│
├── model/ # Trained ML models
│ ├── naive_bayes_model.pkl
│ ├── naive_bayes_model_v2.pkl
│ ├── random_forest_model.pkl
│ ├── random_forest_model_v2.pkl
│ ├── svm_model.pkl
│ └── svm_model_v2.pkl
│
├── vectorizer/
│ └── tfidf_vectorizer.pkl
│
└── emotion_env/

---

## 🚀 Installation

1. **Clone repository:**

```bash
git clone https://github.com/hilmansw/streamlit-emotion-detector.git
cd streamlit-emotion-detector
```

2. Create virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate environment:

- Windows: venv\Scripts\activate
- Linux/Mac: source venv/bin/activate

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

Run the following command:

```bash
streamlit run app.py
```
