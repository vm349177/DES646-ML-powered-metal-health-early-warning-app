import os
import sounddevice as sd
import wavio
import numpy as np
import tempfile
import whisper
import torch
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import soundfile as sf
# ==============================
# CONFIGURATION
# ==============================
MODEL_DIR = "./Mental-Health-Sentiment-Analysis-using-Deep-Learning-main/output_fast_training"      # your trained RoBERTa model folder
MAX_LEN = 96
SAMPLE_RATE = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id_to_label = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# ==============================
# LOAD MODELS
# ==============================
print("🔄 Loading models...")
try:
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    sentiment_model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    print("✅ Loaded fine-tuned RoBERTa model.")
except Exception as e:
    print(f"⚠️ Could not load fine-tuned model, loading base 'roberta-base': {e}")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    sentiment_model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)

whisper_model = whisper.load_model("tiny.en")
print("✅ Whisper speech-to-text model loaded.\n")

# ==============================
# RECORD AUDIO (Manual Start/Stop)
# ==============================
def record_audio_interactive(fs=SAMPLE_RATE):
    print("\n🎙️ Press ENTER to start recording.")
    input()
    print("Recording... (press ENTER again to stop)")
    recording = []
    stream = sd.InputStream(samplerate=fs, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            block, overflowed = stream.read(1024)
            recording.append(block)
            if os.name == 'nt':
                import msvcrt
                if msvcrt.kbhit() and msvcrt.getch() == b'\r':  # ENTER key
                    break
            else:
                import sys, select
                i, o, e = select.select([sys.stdin], [], [], 0)
                if i:
                    sys.stdin.readline()
                    break
    finally:
        stream.stop()
        stream.close()

    audio = np.concatenate(recording, axis=0)
    print("✅ Recording stopped.\n")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavio.write(tmp.name, audio, fs, sampwidth=2)
    return tmp.name

# ==============================
# TRANSCRIBE AUDIO
# ==============================
# ==============================
def transcribe_audio(audio_path):
    print("🧠 Transcribing speech with Whisper (no FFmpeg)...")

    # Load raw audio directly from WAV
    audio_data, sample_rate = sf.read(audio_path)

    # Convert stereo → mono if needed
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if not 16kHz
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Convert dtype to float32 (important)
    audio_data = audio_data.astype(np.float32)

    # Whisper expects 16kHz float32 waveform directly
    result = whisper_model.transcribe(audio_data, fp16=False)
    text = result["text"].strip()

    print(f"🗣️ Transcribed Text: {text}\n")
    return text
# ==============================
# PREDICT SENTIMENT
# ==============================
def predict_sentiment(text):
    sentiment_model.eval()
    enc = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sentiment_model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    pred_label = id_to_label.get(pred, "Unknown")
    probs_dict = {id_to_label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
    return pred_label, probs_dict

# ==============================
# MAIN PIPELINE
# ==============================
def voice_to_sentiment():
    audio_file = record_audio_interactive()
    text = transcribe_audio(audio_file)
    if not text:
        print("⚠️ No speech detected.")
        return

    print("💬 Predicting sentiment...\n")
    label, probs = predict_sentiment(text)
    print("🧩 FINAL RESULT")
    print("🗣️ Transcribed:", text)
    print("🎯 Sentiment:", label)
    print("📊 Probabilities:")
    for k, v in probs.items():
        print(f"  - {k:<15}: {v:.4f}")
    print("\n✅ Done!\n")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    voice_to_sentiment()
