import streamlit as st
import tensorflow as tf
import pickle

# === Constants ===
MAX_LEN = 200
MAX_WORDS = 10000

# === App UI ===
st.set_page_config(page_title="IMDb Sentiment Classifier")
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("- BiLSTM (custom-built + weights loaded)")

st.title("🎬 IMDb Sentiment Classifier")
st.markdown("Enter a movie review below:")

# === Rebuild Model Structure ===
def build_bilstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Load model and tokenizer
try:
    model_lstm = build_bilstm_model()
    model_lstm.load_weights("bilstm_weights.h5")

    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)

    st.success("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or tokenizer: {e}")

# === Input + Predict ===
text = st.text_area("✍️ Enter your movie review:")

if st.button("Predict"):
    try:
        seq = lstm_tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
        pred = model_lstm.predict(padded)[0][0]
        label = "😊 Positive" if pred > 0.5 else "😠 Negative"
        st.subheader(f"Prediction: {label}")
        st.caption(f"Confidence: {pred:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
import streamlit as st
import tensorflow as tf
import pickle

# === Constants ===
MAX_LEN = 200
MAX_WORDS = 10000

# === App UI ===
st.set_page_config(page_title="IMDb Sentiment Classifier")
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("- BiLSTM (custom-built + weights loaded)")

st.title("🎬 IMDb Sentiment Classifier")
st.markdown("Enter a movie review below:")

# === Rebuild Model Structure ===
def build_bilstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Load model and tokenizer
try:
    model_lstm = build_bilstm_model()
    model_lstm.load_weights("bilstm_weights.h5")

    with open("bilstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)

    st.success("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or tokenizer: {e}")

# === Input + Predict ===
text = st.text_area("✍️ Enter your movie review:")

if st.button("Predict"):
    try:
        seq = lstm_tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
        pred = model_lstm.predict(padded)[0][0]
        label = "😊 Positive" if pred > 0.5 else "😠 Negative"
        st.subheader(f"Prediction: {label}")
        st.caption(f"Confidence: {pred:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
