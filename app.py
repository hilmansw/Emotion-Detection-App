import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Load Model, Encoder, Vectorizer
# ==========================

# Load model
svm_sentiment = joblib.load("model/svm_model.pkl")
svm_emotion = joblib.load("model/svm_model_v2.pkl")

# Encoder
label_encoder_sentiment = joblib.load("encoder/label_encoder_sentiment.pkl")
label_encoder_emotion = joblib.load("encoder/label_encoder_emotion.pkl")

# Vectorizer
tfidf_vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")

# ==========================
# Stopwords
# ==========================
stop_words = set(stopwords.words('indonesian'))

# ==========================
# Preprocessing
# ==========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1+', r'\1\1', text) 
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
    text = ' '.join(text.split()) 
    tokens = word_tokenize(text) 
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ==========================
# Streamlit UI
# ==========================
st.title("Emotion Detection App")
st.write("**Made by:** Hilman Singgih Wicaksana, S.Kom., M.Kom.")

user_input = st.text_area("Tulis kalimat di sini:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Harap masukkan kalimat terlebih dahulu.")
    else:
        processed_text = preprocess_text(user_input)

        # --- Prediksi Sentimen ---
        vec_sentiment = tfidf_vectorizer.transform([processed_text])
        pred_sentiment = svm_sentiment.predict(vec_sentiment)[0]
        label_sentiment = label_encoder_sentiment.inverse_transform([pred_sentiment])[0]

        # --- Prediksi Emosi ---
        vec_emotion = tfidf_vectorizer.transform([processed_text])
        pred_emotion = svm_emotion.predict(vec_emotion)[0]
        label_emotion = label_encoder_emotion.inverse_transform([pred_emotion])[0]

        # --- Output ---
        st.subheader("Hasil Prediksi")
        st.write(f"**Sentimen:** {label_sentiment}")
        st.write(f"**Emosi:** {label_emotion}")
