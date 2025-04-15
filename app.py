import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# === Constants ===
MAX_LEN = 200
MAX_WORDS = 10000

# === App UI ===
st.set_page_config(page_title="IMDb Sentiment Classifier", layout="centered")
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("Using a BiLSTM model (Keras 3 compatible)")
st.title("🎬 IMDb Sentiment Classifier (BiLSTM Only)")
st.markdown("Enter a movie review below to get the sentiment prediction:")

# === Rebuild the same BiLSTM architecture ===
def build_bilstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# === Load model weights and tokenizer ===
model_lstm = None
lstm_tokenizer = None

try:
    # Rebuild and build model
    model_lstm = build_bilstm_model()
    model_lstm.build(input_shape=(None, MAX_LEN))
    model_lstm.load_weights("bilstm_weights.h5")

    # Load tokenizer from JSON (new, Keras 3-compatible way)
    with open("tokenizer.json", "r") as f:
        token_json = json.load(f)  # Read the tokenizer JSON data from the file
        
        if not token_json:  # Check if tokenizer JSON data is empty or invalid
            raise ValueError("Tokenizer JSON is empty or invalid.")
        
        lstm_tokenizer = tokenizer_from_json(token_json)  # Load tokenizer from JSON data

    st.success("✅ Model and tokenizer loaded successfully!")

except Exception as e:
    st.error(f"❌ Error loading model or tokenizer: {e}")

# === Text Input and Prediction ===
text = st.text_area("✍️ Enter your movie review:")

if st.button("Predict"):
    if lstm_tokenizer is None:
        st.error("❌ Tokenizer not loaded correctly. Please reload the page.")
    else:
        try:
            # Preprocess the text
            sequence = lstm_tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)

            # Predict
            prediction = model_lstm.predict(padded)[0][0]
            label = "😊 Positive" if prediction > 0.5 else "😠 Negative"

            # Output
            st.subheader(f"Prediction: {label}")
            st.caption(f"Confidence: {prediction:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
