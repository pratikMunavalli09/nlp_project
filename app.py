# app.py
import os
import streamlit as st
import tensorflow as tf # Now import TensorFlow AFTER setting the environment variable
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- Configuration ---
MODEL_PATH = "bilstm_model.keras" # Using the .keras format
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"
MAX_LEN = 200

# --- NLTK Stopwords Download ---
@st.cache_resource
def download_nltk_data():
    """Downloads NLTK 'stopwords' data, caching the attempt."""
    try:
        nltk.download('stopwords')
        st.write("NLTK 'stopwords' data check/download successful.")
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

nltk_download_successful = download_nltk_data()

# --- Load Stopwords ---
stop_words = set()
if nltk_download_successful:
    try:
        stop_words = set(stopwords.words('english'))
        st.write(f"Loaded {len(stop_words)} English stopwords.")
    except LookupError:
        st.error("Stopwords not found after download attempt. Text cleaning might be affected.")
    except Exception as e:
        st.error(f"An unexpected error occurred loading stopwords: {e}")
else:
    st.warning("NLTK data download failed. Proceeding without stopwords removal.")

# --- Text Cleaning Function ---
def clean_text(text):
    """Cleans input text: lowercase, remove HTML, punctuation, numbers, stopwords."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    if stop_words:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    else:
        text = ' '.join(text.split())
    return text

# --- Load Model and Tokenizer (Cached) ---
@st.cache_resource
def load_resources():
    """Loads the Keras model and tokenizer from files, caching the result."""
    model = None
    tokenizer = None
    model_loaded = False
    tokenizer_loaded = False

    # --- Check for GPU (for debugging purposes AFTER forcing CPU) ---
    # This should now report no GPUs available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.warning(f"TensorFlow still sees GPUs: {gpus} - CUDA_VISIBLE_DEVICES might not have taken effect early enough.")
        else:
            st.write("TensorFlow correctly sees no GPUs available.")
    except Exception as e:
        st.write(f"Could not check for GPUs: {e}")
    # --- End GPU Check ---


    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'")
    else:
        try:
            # Load Model - this should now only use CPU
            model = tf.keras.models.load_model(MODEL_PATH)
            st.write(f"Model loaded successfully from {MODEL_PATH}")
            model_loaded = True
        except Exception as e:
            # Provide more detail if loading fails
            st.error(f"Error loading Keras model from {MODEL_PATH}: {e}")
            st.error(f"TensorFlow version being used: {tf.__version__}") # Log TF version


    if not os.path.exists(TOKENIZER_CONFIG_PATH):
        st.error(f"Tokenizer config file not found at '{TOKENIZER_CONFIG_PATH}'")
    else:
        try:
            with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
                tokenizer_json = f.read()
                tokenizer = tokenizer_from_json(tokenizer_json)
            st.write(f"Tokenizer loaded successfully from {TOKENIZER_CONFIG_PATH}")
            tokenizer_loaded = True
        except Exception as e:
            st.error(f"Error loading tokenizer from {TOKENIZER_CONFIG_PATH}: {e}")

    return model if model_loaded else None, tokenizer if tokenizer_loaded else None

# --- Load resources when the script runs ---
model_lstm, lstm_tokenizer = load_resources()

# --- Streamlit App Interface ---
st.title("🎬 IMDb Sentiment Classifier (BiLSTM)")
st.markdown("Enter a movie review below to classify its sentiment.")

review_text = st.text_area("✍️ Enter your review here:", height=150, placeholder="e.g., 'This movie was fantastic!' or 'A complete waste of time.'")

if st.button("Predict Sentiment"):
    if model_lstm is not None and lstm_tokenizer is not None:
        if review_text and review_text.strip():
            cleaned_review = clean_text(review_text)
            # st.write(f"Cleaned Text: '{cleaned_review[:100]}...'") # Optional debug

            try:
                sequence = lstm_tokenizer.texts_to_sequences([cleaned_review])
                if not sequence or not sequence[0]:
                     st.warning("After cleaning, the review text is empty or contains only unknown words. Cannot predict.")
                else:
                    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
                    prediction = model_lstm.predict(padded_sequence)[0][0]
                    sentiment = "😊 Positive" if prediction > 0.5 else "😠 Negative"
                    confidence = prediction if prediction > 0.5 else 1 - prediction

                    st.subheader(f"Predicted Sentiment: {sentiment}")
                    st.caption(f"Confidence: {confidence:.2f} (Raw Score: {prediction:.4f})")
                    if sentiment == "😊 Positive":
                        st.progress(prediction)
                    else:
                        st.progress(1 - prediction)

            except Exception as e:
                 st.error(f"An error occurred during tokenization or prediction: {e}")

        else:
            st.warning("Please enter a review text before predicting.")
    else:
        st.error("Model or tokenizer failed to load. Cannot predict. Please check the deployment logs or ensure model/tokenizer files are present.")
        st.info(f"Ensure '{MODEL_PATH}' and '{TOKENIZER_CONFIG_PATH}' are present in the repository root.")

# --- Sidebar Information ---
st.sidebar.title("ℹ️ Model Info")
st.sidebar.info(f"**Model Type:** Bidirectional LSTM (BiLSTM)\n\n"
                f"**Max Sequence Length:** {MAX_LEN}\n\n"
                f"**Vocabulary Size:** {lstm_tokenizer.num_words if lstm_tokenizer else 'N/A'}")
st.sidebar.markdown("---")
st.sidebar.subheader("Resource Status")
if model_lstm: st.sidebar.success("✅ Model Loaded")
else: st.sidebar.error("❌ Model Load Failed")
if lstm_tokenizer: st.sidebar.success("✅ Tokenizer Loaded")
else: st.sidebar.error("❌ Tokenizer Load Failed")
if nltk_download_successful and stop_words: st.sidebar.success("✅ NLTK Stopwords Ready")
elif nltk_download_successful: st.sidebar.warning("⚠️ NLTK Downloaded, but Stopwords Failed to Load")
else: st.sidebar.error("❌ NLTK Stopwords Download Failed")
