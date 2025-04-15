import streamlit as st
import tensorflow as tf
# import pickle # No longer needed for tokenizer
import json # Import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json # Import the loader function
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "bilstm_model.h5"
# TOKENIZER_PATH = "bilstm_tokenizer.pkl" # Old path
TOKENIZER_CONFIG_PATH = "tokenizer_config.json" # New path for JSON config
MAX_LEN = 200

# --- NLTK Stopwords Download ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
download_nltk_data()
stop_words = set(stopwords.words('english'))

# --- Text Cleaning Function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- Load Model and Tokenizer (Updated) ---
@st.cache_resource
def load_resources():
    model = None
    tokenizer = None
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at '{MODEL_PATH}'")
            return None, None # Return None for both
        if not os.path.exists(TOKENIZER_CONFIG_PATH):
            st.error(f"Tokenizer config file not found at '{TOKENIZER_CONFIG_PATH}'")
            return None, None # Return None for both

        # Load Model
        model = tf.keras.models.load_model(MODEL_PATH)
        st.write(f"Model loaded successfully from {MODEL_PATH}") # Debug message

        # Load Tokenizer from JSON config
        with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
        st.write(f"Tokenizer loaded successfully from {TOKENIZER_CONFIG_PATH}") # Debug message

        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading resources: {e}")
        # Print specifics if possible
        if model is None:
            st.error("Failed during model loading.")
        if tokenizer is None and model is not None:
             st.error("Failed during tokenizer loading (after model load).")
        return None, None # Ensure None is returned on error

model_lstm, lstm_tokenizer = load_resources()

# --- Streamlit App Interface ---
st.title("🎬 IMDb Sentiment Classifier (BiLSTM)")
st.markdown("Enter a movie review below to classify its sentiment.")

review_text = st.text_area("✍️ Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if model_lstm is not None and lstm_tokenizer is not None:
        if review_text:
            # 1. Clean the input text
            cleaned_review = clean_text(review_text)

            # 2. Tokenize and pad the sequence
            try:
                # Use the loaded tokenizer
                sequence = lstm_tokenizer.texts_to_sequences([cleaned_review])
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

                # 3. Make prediction
                prediction = model_lstm.predict(padded_sequence)[0][0]
                sentiment = "😊 Positive" if prediction > 0.5 else "😠 Negative"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                # 4. Display result
                st.subheader(f"Predicted Sentiment: {sentiment}")
                st.caption(f"Confidence: {confidence:.2f} (Raw Score: {prediction:.4f})")

            except Exception as e:
                 st.error(f"An error occurred during tokenization or prediction: {e}")
                 # st.error(f"Cleaned text was: '{cleaned_review}'") # Optional: uncomment for debugging

        else:
            st.warning("Please enter a review before predicting.")
    else:
        st.error("Model or tokenizer failed to load. Cannot predict. Please check the deployment logs.")
        st.info("Ensure 'bilstm_model.h5' and 'tokenizer_config.json' are present in the repository root.")


st.sidebar.info(f"Model: BiLSTM\nMax Sequence Length: {MAX_LEN}")
# Add check status in sidebar
if model_lstm and lstm_tokenizer:
    st.sidebar.success("✅ Model & Tokenizer Loaded")
else:
    st.sidebar.error("❌ Model/Tokenizer Load Failed")
