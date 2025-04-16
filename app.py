import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from textblob import TextBlob

st.set_page_config(page_title="IMDb Sentiment Classifier")

# --- Config ---
MAX_LEN = 200
MAX_WORDS = 10000
MODEL_WEIGHTS_PATH = "bilstm_weights.h5"
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"

# --- Download NLTK Stopwords ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords')
        return set(stopwords.words('english'))
    except Exception as e:
        st.error(f"Error downloading stopwords: {e}")
        return set()

stop_words = download_nltk_data()

# --- Text Cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# --- Define BiLSTM Model ---
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
    try:
        model.build(input_shape=(None, MAX_LEN))
        model.load_weights(MODEL_WEIGHTS_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        model = None

    tokenizer = None
    try:
        with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(f.read())
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        tokenizer = None

    return model, tokenizer

model_lstm, lstm_tokenizer = load_resources()

# --- Streamlit UI ---
# st.set_page_config(page_title="IMDb Sentiment Classifier")
st.title("🎬 IMDb Sentiment Classifier BiLSTM")
st.markdown("Enter a movie review to classify its sentiment.")

text_input = st.text_area("✍️ Your Review:", height=150, placeholder="Example: This movie was absolutely amazing!")

# --- Predict Sentiment ---
if st.button("Predict Sentiment"):
    # --- TextBlob Analysis ---
    blob = TextBlob(text_input)
    tb_polarity = blob.sentiment.polarity
    tb_sentiment = "😊 Positive" if tb_polarity > 0 else "😠 Negative"
    # tb_sentiment = "😊 Positive" if tb_polarity > 0 else ("😠 Negative" if tb_polarity < 0 else "😐 Neutral")
    
    # --- Output TextBlob Result ---
    st.markdown(f"Predicted Sentiment")
    st.write(f"Polarity Score: `{tb_polarity:.2f}`")
    st.write(f"Predicted Sentiment: **{tb_sentiment}**")
    
    
    if model_lstm is None or lstm_tokenizer is None:
        st.error("Model or tokenizer failed to load.")
    elif not text_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(text_input)
        sequence = lstm_tokenizer.texts_to_sequences([cleaned])
        if not sequence or not sequence[0]:
            st.warning("The cleaned input is empty or contains only unknown words.")
        else:
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
            prediction = model_lstm.predict(padded)[0][0]

            # ✅ Fix: Properly interpret raw prediction
            sentiment = "😊 Positive" if prediction >= 0.5 else "😠 Negative"
            confidence = prediction if prediction >= 0.5 else 1 - prediction


            # ✅ Output
            # st.text(f"Raw prediction score: {prediction:.4f}")
            # st.subheader(f"Predicted Sentiment: {sentiment}")
            # st.caption(f"Confidence: {confidence:.2f}")
            # st.progress(confidence)


# --- Sidebar Info ---
st.sidebar.title("ℹ️ Model Info")
st.sidebar.info(f"**Model Type**: BiLSTM (Keras 3)\n"
                f"**Max Sequence Length**: {MAX_LEN}\n"
                f"**Vocabulary Size**: {MAX_WORDS}")
st.sidebar.markdown("---")
st.sidebar.subheader("Status")
st.sidebar.success("✅ Model Loaded" if model_lstm else "❌ Model Failed")
st.sidebar.success("✅ Tokenizer Loaded" if lstm_tokenizer else "❌ Tokenizer Failed")
st.sidebar.success("✅ Stopwords Loaded" if stop_words else "⚠️ Not Loaded")
