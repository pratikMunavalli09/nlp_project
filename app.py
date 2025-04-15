import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os # Import os to check file paths

# --- Configuration ---
MODEL_PATH = "bilstm_model.h5"
TOKENIZER_PATH = "bilstm_tokenizer.pkl"
MAX_LEN = 200 # Make sure this matches the MAX_LEN used during training

# --- NLTK Stopwords Download (run once) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
download_nltk_data()
stop_words = set(stopwords.words('english'))

# --- Text Cleaning Function (same as in training) ---
def clean_text(text):
    if not isinstance(text, str): # Add check for non-string input
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- Load Model and Tokenizer (Cached) ---
@st.cache_resource # Cache the loaded model and tokenizer
def load_resources():
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None, None
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer file not found at {TOKENIZER_PATH}")
        return None, None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

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
                sequence = lstm_tokenizer.texts_to_sequences([cleaned_review])
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

                # 3. Make prediction
                prediction = model_lstm.predict(padded_sequence)[0][0] # Get the probability from the output array
                sentiment = "😊 Positive" if prediction > 0.5 else "😠 Negative"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                # 4. Display result
                st.subheader(f"Predicted Sentiment: {sentiment}")
                st.caption(f"Confidence: {confidence:.2f} (Raw Score: {prediction:.4f})")

            except Exception as e:
                 st.error(f"An error occurred during prediction: {e}")
                 st.error(f"Cleaned text was: '{cleaned_review}'") # Help debugging

        else:
            st.warning("Please enter a review.")
    else:
        st.error("Model or tokenizer could not be loaded. Please check the logs.")

st.sidebar.info(f"Model: BiLSTM\nMax Sequence Length: {MAX_LEN}")
