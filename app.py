import streamlit as st
import tensorflow as tf
import json
import re
import string
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Config ---
MAX_LEN = 200
MAX_WORDS = 10000
MODEL_WEIGHTS_PATH = "bilstm_weights.h5"
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"

# --- Clean Text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# --- Build BiLSTM ---
def build_bilstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# --- Load Model & Tokenizer ---
@st.cache_resource
def load_resources():
    model = build_bilstm_model()
    model.build(input_shape=(None, MAX_LEN))
    model.load_weights(MODEL_WEIGHTS_PATH)

    with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    return model, tokenizer

# --- Initialize ---
st.set_page_config(page_title="IMDb Sentiment Classifier")
model, tokenizer = load_resources()

# --- Streamlit UI ---
st.title("🎬 IMDb Sentiment Classifier (BiLSTM)")
st.markdown("Enter your own movie review to see the sentiment prediction:")

review = st.text_area("✍️ Your Review", placeholder="e.g., This movie was amazing!")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

        prediction = model.predict(padded)[0][0]
        sentiment = "😊 Positive" if prediction >= 0.5 else "😠 Negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.subheader(f"Predicted Sentiment: {sentiment}")
        st.caption(f"Confidence: {confidence:.2f} (Raw score: {prediction:.4f})")
        st.progress(confidence)
