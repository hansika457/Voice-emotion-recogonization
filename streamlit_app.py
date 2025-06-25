%%writefile app.py
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf

model = load_model("final_model.h5")

emotion_map = {
    0: "angry",
    1: "calm",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised"
}

def extract_mfcc(file_path, max_pad_len=173):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc.T

    return np.expand_dims(mfcc, axis=0).astype(np.float32)

# Streamlit UI
st.title("🎤 Emotion Recognition from Voice")
st.write("Upload a `.wav` audio file to detect the emotion.")

uploaded_file = st.file_uploader("Upload audio", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_mfcc("temp.wav")
    st.write(f"Extracted MFCC shape: {features.shape}")

    prediction = model.predict(features)
    predicted_emotion = emotion_map[np.argmax(prediction)]

    st.success(f" Predicted Emotion: **{predicted_emotion.upper()}**")
  import threading
import time
from pyngrok import ngrok
import os

from google.colab import userdata

ngrok_authtoken = os.environ.get('NGROK_AUTHTOKEN', '2yxWUVcIn0vPsUtptideJUziPEY_6DFmNqSgxaedcMpVGLrxu')
ngrok.set_auth_token(ngrok_authtoken)

def run_app():
    !streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false

thread = threading.Thread(target=run_app)
thread.start()

time.sleep(2)

public_url = ngrok.connect(8501, "http")
print(f" Streamlit app is live at: {public_url}")
