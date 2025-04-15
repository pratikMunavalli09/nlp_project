import streamlit as st
import tensorflow as tf
import pickle

# === App Configuration ===
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("Currently using:")
st.sidebar.markdown("- ✅ BiLSTM (Keras Model)")

MAX_LEN = 200

st.title("🎬 IMDb Sentiment Classifier (BiLSTM Only)")
st.markdown("Enter a movie review below to get the prediction using the BiLSTM model.")

# === Load BiLSTM model and tokenizer ===
try:
    model_lstm = tf.keras.models.load_model("bilstm_model.h5")
    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading BiLSTM model/tokenizer: {e}")

# === Input Area ===
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
