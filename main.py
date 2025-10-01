import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import re
import os

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", str(text))
    text = text.lower()
    return text

# ---------- Load Dataset ----------
def load_balanced_dataset():
    # Sentiment140 dataset format
    df = pd.read_csv("balanced_dataset.csv", encoding="latin-1", header=None)
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]

    # Convert sentiment 4 → 1 (positive), keep 0 as negative
    df['sentiment'] = df['sentiment'].replace(4, 1)

    return df

# ---------- Train Model ----------
def train_model(df):
    df['clean_text'] = df['text'].apply(clean_text)
    X = df['clean_text']
    y = df['sentiment']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer, acc, f1

# ---------- Predict Function ----------
def predict_sentiment(text, model, vectorizer):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)
    return "Positive 😀" if pred[0] == 1 else "Negative 😡"

# ---------- Streamlit UI ----------
st.title("Sentiment Analysis Web App (ML Minor Project)")

# Load or Train Model
model, vectorizer = None, None
if os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    df = load_dataset()
    model, vectorizer, acc, f1 = train_model(df)
    st.write(f"✅ Model trained. Accuracy: {acc:.2f}, F1 score: {f1:.2f}")

# User Input
st.subheader("Predict Sentiment for Text/Tweet")
user_input = st.text_area("Enter text or tweet here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        sentiment = predict_sentiment(user_input, model, vectorizer)
        st.success(f"Predicted Sentiment: **{sentiment}**")

# Retrain Option
st.markdown("---")
st.subheader("Retrain Model (optional)")
if st.button("Retrain Model"):
    df = load_dataset()
    model, vectorizer, acc, f1 = train_model(df)
    st.info(f"🔄 Model retrained! Accuracy: {acc:.2f}, F1 score: {f1:.2f}")

st.markdown("""
**How to use:**  
- Enter any text or tweet above and click 'Predict Sentiment'.  
- To retrain, click 'Retrain Model' (uses dataset.csv in repo).
""")
