import streamlit as st
import tempfile
import io
import numpy as np
import soundfile as sf
import librosa
import whisper
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from audio_recorder_streamlit import audio_recorder
import plotly.express as px  # <-- added for charts

# =========================
# CONFIGURATION
# =========================
MODEL_DIR = "./Mental-Health-Sentiment-Analysis-using-Deep-Learning-main/output_fast_training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_LEN = 96

id_to_label = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny.en")
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
        sentiment_model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        st.success("✅ Loaded fine-tuned RoBERTa model.")
    except Exception as e:
        st.warning(f"⚠️ Could not load fine-tuned model, loading base roberta-base: {e}")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        sentiment_model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(DEVICE)
    return whisper_model, tokenizer, sentiment_model

whisper_model, tokenizer, sentiment_model = load_models()

# =========================
# HELPER FUNCTIONS
# =========================
def wav_bytes_to_float32_np(wav_bytes):
    """Convert WAV bytes to a float32 numpy array."""
    bio = io.BytesIO(wav_bytes)
    data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data.astype(np.float32)

def transcribe_with_whisper_from_bytes(wav_bytes):
    """Transcribe recorded audio using Whisper."""
    audio_np = wav_bytes_to_float32_np(wav_bytes)
    result = whisper_model.transcribe(audio_np, fp16=False)
    return result.get("text", "").strip()

def predict_sentiment_from_text(text):
    """Run the sentiment classification model."""
    sentiment_model.eval()
    enc = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = sentiment_model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    pred_label = id_to_label.get(pred, "Unknown")
    probs_dict = {id_to_label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
    return pred_label, probs_dict

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Voice-to-Sentiment", page_icon="🎙", layout="centered")
st.title("🎧 Voice-to-Sentiment Analyzer with Visualization")
st.write("Click the **Record** button below, speak, and the app will transcribe your voice and analyze its sentiment.")

# Record audio directly in the browser
wav_bytes = audio_recorder()

if wav_bytes is not None:
    st.success("✅ Audio recorded in browser and received on server.")
    st.audio(wav_bytes, format="audio/wav")

    with st.spinner("🧠 Transcribing speech (Whisper)..."):
        try:
            transcript = transcribe_with_whisper_from_bytes(wav_bytes)
        except Exception as e:
            st.error(f"❌ Whisper transcription failed: {e}")
            transcript = ""

    st.markdown("### 🗣 Transcribed Text")
    st.write(transcript if transcript else "_No speech detected._")

    if transcript:
        with st.spinner("🔍 Predicting sentiment..."):
            label, probs = predict_sentiment_from_text(transcript)

        st.markdown(f"### 🎯 Predicted Sentiment: **{label}**")

        # =========================
        # VISUALIZATION
        # =========================
        st.subheader("📊 Sentiment Probability Visualization")

        # Convert probabilities to DataFrame
        import pandas as pd
        df = pd.DataFrame(list(probs.items()), columns=["Sentiment", "Probability"])

        # Tabs for visualization options
        chart_tab, pie_tab = st.tabs(["📈 Bar Chart", "🥧 Pie Chart"])

        with chart_tab:
            fig_bar = px.bar(
                df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                title="Sentiment Probability Distribution",
                text=[f"{p:.2f}" for p in df["Probability"]],
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, width="stretch")

        with pie_tab:
            fig_pie = px.pie(
                df,
                names="Sentiment",
                values="Probability",
                title="Sentiment Class Probabilities",
                hole=0.3,
            )
            st.plotly_chart(fig_pie, width="stretch")
