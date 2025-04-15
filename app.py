import streamlit as st
import tensorflow as tf
import pickle

# === App Sidebar Info ===
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("Currently using:")
st.sidebar.markdown("- ✅ BiLSTM (Keras Model)")

MAX_LEN = 200

st.title("🎬 IMDb Sentiment Classifier (BiLSTM)")
st.markdown("Enter a movie review below and get a sentiment prediction from the BiLSTM model.")

# === Load BiLSTM model and tokenizer ===
try:
    # Load model with compile=False to avoid time_major error
    model_lstm = tf.keras.models.load_model("bilstm_model.h5", compile=False)

    # Load tokenizer
    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)

    st.success("✅ BiLSTM model and tokenizer loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading BiLSTM model/tokenizer: {e}")

# === Input Box ===
text = st.text_area("✍️ Enter your review here:")

# === Prediction Logic ===
if st.button("Predict with BiLSTM"):
    try:
        # Preprocess and predict
        sequence = lstm_tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)
        prediction = model_lstm.predict(padded)[0][0]

        # Classify sentiment
        label = "😊 Positive" if prediction > 0.5 else "😠 Negative"

        # Display result
        st.subheader(f"BiLSTM Prediction: {label}")
        st.caption(f"Confidence: {prediction:.2f}")
    except Exception as e:
        st.error(f"❌ BiLSTM Prediction Error: {e}")
