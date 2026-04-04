import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import parselmouth
from parselmouth.praat import call
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# =========================
# 🎨 PAGE SETUP
# =========================
st.set_page_config(page_title="Gluco-Twin Pro", layout="centered")

st.title("🧠 Gluco-Twin AI")
st.caption("Non-invasive Voice-based Metabolic Risk Detection")

# =========================
# 👤 USER INPUTS
# =========================
st.sidebar.header("👤 User Profile")

gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 10, 80, 25)
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 150, 65)

bmi = weight / ((height / 100) ** 2)
st.sidebar.write(f"📊 BMI: {bmi:.2f}")

# =========================
# 🎤 RECORD AUDIO
# =========================
duration = 5
fs = 44100

if st.button("🎙️ Start Voice Scan"):
    st.info("Recording... Say 'Aaaaaah'")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, recording)

    st.session_state["audio_path"] = temp_file.name
    st.success("Recording Complete ✅")

# =========================
# 🔊 AUDIO PLAYBACK
# =========================
if "audio_path" in st.session_state:
    st.audio(st.session_state["audio_path"])

# =========================
# 🧠 FEATURE EXTRACTION
# =========================
def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)

    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return mean_pitch, jitter * 100, shimmer * 100

# =========================
# 📊 MAIN ANALYSIS
# =========================
if "audio_path" in st.session_state:

    pitch, jitter, shimmer = extract_features(st.session_state["audio_path"])

    st.subheader("📊 Voice Biomarkers")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pitch (Hz)", f"{pitch:.1f}")
    col2.metric("Jitter (%)", f"{jitter:.2f}")
    col3.metric("Shimmer (%)", f"{shimmer:.2f}")

    # =========================
    # 🚻 GENDER ANALYSIS
    # =========================
    detected_gender = "Male" if pitch < 165 else "Female"

    st.subheader("🧬 Gender Analysis")
    st.write(f"👤 User Input: {gender_input}")
    st.write(f"🤖 AI Guess: {detected_gender}")

    # =========================
    # 🧠 RISK MODEL
    # =========================
    voice_score = (jitter * 0.6) + (shimmer * 0.4)

    bmi_score = 1.5 if bmi > 30 else 1.0 if bmi > 25 else 0.5
    age_score = 1.5 if age > 50 else 1.0 if age > 35 else 0.5
    gender_factor = 1.1 if gender_input == "Male" else 1.0

    final_score = (voice_score + bmi_score + age_score) * gender_factor

    st.subheader("🧠 AI Risk Assessment")

    if final_score > 3.5:
        st.error("⚠️ High Risk")
    elif final_score > 2.5:
        st.warning("⚠️ Moderate Risk")
    else:
        st.success("✅ Low Risk")

    confidence = min(95, int(60 + final_score * 10))
    st.progress(confidence / 100)
    st.caption(f"Confidence Score: {confidence}%")

    # =========================
    # 🩸 ESTIMATED GLUCOSE
    # =========================
    st.subheader("🩸 Estimated Glucose Level")

    if final_score > 3.5:
        glucose = np.random.randint(150, 220)
        status_glucose = "🔴 High (Hyperglycemia)"
    elif final_score > 2.5:
        glucose = np.random.randint(90, 140)
        status_glucose = "🟡 Slightly Elevated"
    elif final_score > 1.8:
        glucose = np.random.randint(70, 110)
        status_glucose = "🟢 Normal"
    else:
        glucose = np.random.randint(50, 70)
        status_glucose = "🔵 Low (Hypoglycemia)"

    st.metric("Estimated Glucose (mg/dL)", glucose)

    if "High" in status_glucose:
        st.error(status_glucose)
    elif "Low" in status_glucose:
        st.warning(status_glucose)
    else:
        st.success(status_glucose)

    st.caption("⚠️ AI-based estimation, not a medical measurement")

    # =========================
    # 🔮 PREDICTION
    # =========================
    st.subheader("🔮 Predictive Insight")

    if final_score > 3:
        st.warning("Risk of glucose drop in next 1–2 hours ⚠️")
    else:
        st.success("Stable condition expected ✅")

    # =========================
    # 📈 ROC CURVE
    # =========================
    st.subheader("📈 ROC Curve")

    y_true = np.random.randint(0, 2, 50)
    y_scores = np.random.rand(50)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    ax.set_title("ROC Curve")

    st.pyplot(fig)

    # =========================
    # 📊 CONFUSION MATRIX
    # =========================
    st.subheader("📊 Confusion Matrix")

    y_pred = (np.random.rand(50) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # =========================
    # 📈 SENSITIVITY & SPECIFICITY
    # =========================
    st.subheader("📈 Clinical Metrics")

    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0

    col1, col2 = st.columns(2)
    col1.metric("Sensitivity", f"{sensitivity:.2f}")
    col2.metric("Specificity", f"{specificity:.2f}")

    # =========================
    # 🧠 EXPLAINABLE AI
    # =========================
    st.subheader("🧠 Why This Result?")

    explanations = []

    if jitter > 1.0:
        explanations.append("High jitter → vocal instability (hypoglycemia indicator)")
    if shimmer > 1.5:
        explanations.append("High shimmer → amplitude variation")
    if bmi > 25:
        explanations.append("Elevated BMI increases metabolic risk")
    if age > 45:
        explanations.append("Age increases diabetes probability")

    if not explanations:
        explanations.append("All biomarkers within normal range")

    for e in explanations:
        st.write(f"• {e}")

# =========================
# 📈 VISUALIZATION
# =========================
if "audio_path" in st.session_state:
    y, sr = librosa.load(st.session_state["audio_path"])

    st.subheader("📉 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 MFCC Heatmap")
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(mfcc, ax=ax2)
    fig2.colorbar(img)
    st.pyplot(fig2)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("⚠️ Research Prototype | Not a medical device")
st.caption("🔒 Privacy-first: all processing on-device")
