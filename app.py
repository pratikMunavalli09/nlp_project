import streamlit as st
import tensorflow as tf
import pickle
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# === Load BiLSTM Model ===
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("Using two models:")
st.sidebar.markdown("- BiLSTM (Keras)")
st.sidebar.markdown("- DistilBERT (Hugging Face)")

MAX_LEN = 200

st.title("🎬 IMDb Sentiment Classifier")
st.markdown("Enter a movie review below to see how both models classify it!")

# Load BiLSTM model and tokenizer
try:
    model_lstm = tf.keras.models.load_model("bilstm_model.h5")
    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading BiLSTM model/tokenizer: {e}")

# Load DistilBERT model and tokenizer
try:
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_model")
    bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_model")
except Exception as e:
    st.error(f"Error loading DistilBERT model/tokenizer: {e}")

# Input box
text = st.text_area("✍️ Enter your review here:")

if st.button("Predict with BiLSTM"):
    try:
        sequence = lstm_tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)
        prediction = model_lstm.predict(padded)[0][0]
        label = "😊 Positive" if prediction > 0.5 else "😠 Negative"
        st.subheader(f"BiLSTM Prediction: {label}")
        st.caption(f"Confidence: {prediction:.2f}")
    except Exception as e:
        st.error(f"BiLSTM Error: {e}")

if st.button("Predict with DistilBERT"):
    try:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
            sentiment = "😊 Positive" if pred_label == 1 else "😠 Negative"
        st.subheader(f"DistilBERT Prediction: {sentiment}")
        st.caption(f"Confidence: {confidence:.2f}")
    except Exception as e:
        st.error(f"DistilBERT Error: {e}")
