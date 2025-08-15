import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Stopwords load
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------- Load model & vectorizer ----------
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# ---------- SAME Cleaning function as training ----------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ---------- Streamlit UI ----------
st.title("📰 Fake News Detection App")
st.write("Enter news text below to check if it's real or fake.")

user_input = st.text_area("Paste the news article here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_text)[0]
        proba = model.predict_proba(transformed_text)[0]

        label = "FAKE" if prediction == 1 else "REAL"

        confidence = round(max(proba) * 100, 2)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence}%**")
