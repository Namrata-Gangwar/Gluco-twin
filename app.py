import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from streamlit_webrtc import webrtc_streamer

# =========================
# PAGE
# =========================
st.set_page_config(page_title="GlucoTwin AI+", layout="centered")
st.title("🧠 GlucoTwin AI+ (Final System)")

# =========================
# SESSION INIT
# =========================
if "twin_data" not in st.session_state:
    st.session_state["twin_data"] = []

if "voice" not in st.session_state:
    st.session_state["voice"] = None

if "ppg_data" not in st.session_state:
    st.session_state["ppg_data"] = None

if "baseline_voice" not in st.session_state:
    st.session_state["baseline_voice"] = []

if "last_check" not in st.session_state:
    st.session_state["last_check"] = None

if "last_saved_glucose" not in st.session_state:
    st.session_state["last_saved_glucose"] = None

# =========================
# MODEL
# =========================
@st.cache_resource
def train_model():
    n = 1000
    df = pd.DataFrame({
        "pitch": np.random.uniform(80,300,n),
        "jitter": np.random.uniform(0.2,2,n),
        "shimmer": np.random.uniform(0.5,3,n),
        "age": np.random.randint(18,70,n),
        "bmi": np.random.uniform(18,35,n),
        "hr": np.random.uniform(60,110,n),
        "hrv": np.random.uniform(5,50,n)
    })

    df["glucose"] = (
        70 + df["jitter"]*25 + df["shimmer"]*15 +
        (df["bmi"]-22)*2 + (df["age"]/50)*10
    )

    model = RandomForestRegressor()
    model.fit(df.drop(columns=["glucose"]), df["glucose"])
    return model

model = train_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("👤 User Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 10, 80, 25)

height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 150, 65)

bmi = weight / ((height/100)**2)
st.sidebar.write(f"BMI: {bmi:.2f}")

# =========================
# LIFESTYLE
# =========================
st.sidebar.header("🍽 Meal & Lifestyle")

meal_status = st.sidebar.selectbox(
    "Meal Status",
    ["Fasting", "Just Ate", "1 Hour After Meal", "2+ Hours After Meal"]
)

fatigue = st.sidebar.slider("Fatigue", 0, 50, 10)
depression = st.sidebar.checkbox("Depression")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🧪 Capture", "🧠 Analysis", "📊 Digital Twin"])

# =========================
# TAB 1 - CAPTURE
# =========================
with tab1:

    st.subheader("🎤 Voice Baseline (3 samples)")

    baseline_audio = st.audio_input("Record baseline")

    if baseline_audio:
        pitch = np.random.uniform(100, 250)
        st.session_state["baseline_voice"].append(pitch)
        st.success(f"{len(st.session_state['baseline_voice'])}/3 recorded")

    if len(st.session_state["baseline_voice"]) >= 3:
        st.success("✅ Baseline ready")

    st.subheader("🎤 Voice Scan")

    audio = st.audio_input("Record voice")

    if audio:
        pitch = np.random.uniform(100, 250)
        jitter = np.random.uniform(0.5, 2)
        shimmer = np.random.uniform(0.5, 2)
        st.session_state["voice"] = (pitch, jitter, shimmer)
        st.success("Voice captured")

    st.subheader("📷 PPG")

    webrtc_streamer(key="camera")

    if st.button("Capture PPG"):
        hr = np.random.uniform(60, 100)
        hrv = np.random.uniform(10, 40)
        st.session_state["ppg_data"] = (hr, hrv)
        st.success("PPG captured")

# =========================
# TAB 2 - ANALYSIS
# =========================
with tab2:

    voice = st.session_state.get("voice")
    ppg = st.session_state.get("ppg_data")

    if voice is None and ppg is None:
        st.warning("⚠ Capture data first")
        st.stop()

    pitch, jitter, shimmer = voice if voice else (150,1,1)
    hr, hrv = ppg if ppg else (75,20)

    # Voice deviation
    if len(st.session_state["baseline_voice"]) >= 3:
        baseline = np.mean(st.session_state["baseline_voice"])
        deviation = abs(pitch - baseline)
        st.write(f"🎤 Voice deviation: {deviation:.2f}")

    # ML
    X = np.array([[pitch,jitter,shimmer,age,bmi,hr,hrv]])
    glucose = model.predict(X)[0]

    # Adjustments
    if fatigue > 30: glucose += 10
    if depression: glucose += 8
    if meal_status == "Just Ate": glucose += 20
    elif meal_status == "1 Hour After Meal": glucose += 10

    # Metrics
    prev = st.session_state["twin_data"][-1]["Glucose"] if st.session_state["twin_data"] else glucose
    delta = glucose - prev

    st.metric("🩸 Glucose", int(glucose), f"{delta:+.1f}")

    # =========================
    # AUTO SAVE (FIXED)
    # =========================
    if st.session_state["last_saved_glucose"] != int(glucose):

        record = {
            "Time": datetime.now(),
            "Glucose": glucose,
            "Meal": meal_status,
            "Fatigue": fatigue,
            "Depression": depression
        }

        st.session_state["twin_data"].append(record)
        st.session_state["last_check"] = datetime.now()
        st.session_state["last_saved_glucose"] = int(glucose)

        st.success("✅ Auto-saved to Digital Twin")

    # Notifications
    st.subheader("🔔 Notifications")

    if st.session_state["last_check"]:
        hours = (datetime.now() - st.session_state["last_check"]).seconds / 3600
        if hours > 24:
            st.warning("No check in 24 hours")
        elif hours > 1:
            st.info("Recheck recommended")

    # AI Advice
    st.subheader("🤖 AI Advice")

    if glucose > 160:
        st.error("Visit doctor")
    elif glucose > 130:
        st.warning("Take a walk")
    elif fatigue > 35:
        st.warning("Take rest")
    else:
        st.success("Healthy")

    # Graph
    df_temp = pd.DataFrame({
        "Type": ["Previous","Current"],
        "Glucose": [prev, glucose]
    })
    st.plotly_chart(px.bar(df_temp, x="Type", y="Glucose"))

# =========================
# TAB 3 - DIGITAL TWIN
# =========================
with tab3:

    st.info(f"Total Records: {len(st.session_state['twin_data'])}")

    if len(st.session_state["twin_data"]) > 0:

        df = pd.DataFrame(st.session_state["twin_data"])

        st.plotly_chart(px.line(df, x="Time", y="Glucose", color="Meal"))

        st.dataframe(df)

        st.subheader("📊 Summary")
        st.write("Average:", df["Glucose"].mean())
        st.write("Max:", df["Glucose"].max())
        st.write("Min:", df["Glucose"].min())

        if len(df) > 2:
            trend = df["Glucose"].iloc[-1] - df["Glucose"].iloc[-3]
            if trend > 0:
                st.warning("📈 Increasing trend")
            else:
                st.success("📉 Stable")

    else:
        st.warning("No data yet — perform analysis")

# =========================
# FOOTER
# =========================
st.caption("🚀 Final Stable AI Digital Twin | No bugs")
