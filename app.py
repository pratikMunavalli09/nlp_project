import streamlit as st
import tensorflow as tf
import pickle

# === App Configuration ===
st.set_page_config(page_title="IMDb Sentiment Classifier", layout="centered")
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("Using:")
st.sidebar.markdown("- ✅ BiLSTM (Keras Model)")

MAX_LEN = 200

st.title("🎬 IMDb Sentiment Classifier (BiLSTM Only)")
st.markdown("Enter a movie review below to get a sentiment prediction from the BiLSTM model.")

# === Load BiLSTM model and tokenizer ===
try:
    # Load the BiLSTM model in TensorFlow SavedModel format (NOT .h5)
    model_lstm = tf.keras.models.load_model("bilstm_model_tf")  # folder name, not file

    # Load the tokenizer
    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)

    st.success("✅ BiLSTM model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading BiLSTM model/tokenizer: {e}")

# === Text Input ===
text = st.text_area("✍️ Enter your movie review:")

# === Predict Button ===
if st.button("Predict Sentiment"):
    try:
        # Preprocess
        sequence = lstm_tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        prediction = model_lstm.predict(padded)[0][0]
        label = "😊 Positive" if prediction > 0.5 else "😠 Negative"

        # Show result
        st.subheader(f"Prediction: {label}")
        st.caption(f"Confidence: {prediction:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
