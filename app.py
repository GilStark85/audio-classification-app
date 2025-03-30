import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# Configure the page
st.set_page_config(page_title="Audio Classifier", layout="centered")
st.title("🎧 Audio Classification: Dialogue, Music, or SFX")

# Load the trained model (cached for performance)
@st.cache_resource
def load_model():
    return joblib.load("audio_model.pkl")

model = load_model()

# Extract both mean and std of MFCC features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    return np.hstack([mfccs_mean, mfccs_std])

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

# Process and predict
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("🔍 Extracting features and making prediction..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Extract features
            features = extract_features(tmp_path)

            # Predict with probabilities
            probs = model.predict_proba([features])[0]
            classes = model.classes_
            top_index = np.argmax(probs)
            top_class = classes[top_index]
            confidence = probs[top_index]

            # Display top prediction with confidence
            st.success(f"🎯 Predicted Class: **{top_class.capitalize()}** ({confidence * 100:.2f}% confidence)")

            # Display full class probability breakdown
            st.markdown("### 🔍 Class Probabilities:")
            for cls, prob in zip(classes, probs):
                st.markdown(f"- **{cls.capitalize()}**: {prob * 100:.2f}%")

        except Exception as e:
            st.error(f"❌ Oops! Something went wrong during prediction.\n\nError: {e}")
