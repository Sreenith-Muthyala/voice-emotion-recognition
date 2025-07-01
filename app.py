import streamlit as st
import librosa
import numpy as np
import joblib
import os

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), "../model/emotion_model.pkl")
model = joblib.load(model_path)

# Function to extract MFCC from audio
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed.reshape(1, -1)

# Streamlit app
st.title("🎙️ Voice Emotion Recognition")
st.write("Upload an audio file (.wav) to predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    # Extract features
    features = extract_features("temp.wav")

    # Predict emotion
    prediction = model.predict(features)[0]
    st.markdown(f"### 🎯 Predicted Emotion: **{prediction.capitalize()}**")
