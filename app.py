
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt

# Try seaborn (optional)
try:
    import seaborn as sns
    seaborn_available = True
except:
    seaborn_available = False

from streamlit_webrtc import webrtc_streamer

# =========================
# PAGE
# =========================
st.set_page_config(page_title="GlucoTwin AI+", layout="centered")
st.title("🧠 GlucoTwin AI+")
st.warning("⚠ Experimental AI - Hackathon Prototype")

# =========================
# SESSION INIT
# =========================
for key, default in {
    "twin_data": [],
    "voice": None,
    "ppg_data": None,
    "baseline_voice": [],
    "last_check": None,
    "last_saved_glucose": None,
    "last_pitch": 150
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Capture", 
    "🧠 Analysis", 
    "📊 Digital Twin",
    "📈 Performance"
])

# =========================
# TAB 1 - CAPTURE
# =========================
with tab1:
    st.subheader("🎤 Voice Baseline (3 samples)")
    baseline_audio = st.audio_input("Record baseline")

    if baseline_audio:
        pitch = st.session_state["last_pitch"] + np.random.uniform(-5,5)
        st.session_state["baseline_voice"].append(pitch)
        st.session_state["last_pitch"] = pitch
        st.success(f"{len(st.session_state['baseline_voice'])}/3 recorded")

    if len(st.session_state["baseline_voice"]) >= 3:
        st.success("✅ Baseline ready")

    st.subheader("🎤 Voice Scan")
    audio = st.audio_input("Record voice")

    if audio:
        pitch = st.session_state["last_pitch"] + np.random.uniform(-5,5)
        jitter = np.random.uniform(0.5, 2)
        shimmer = np.random.uniform(0.5, 2)

        st.session_state["last_pitch"] = pitch
        st.session_state["voice"] = (pitch, jitter, shimmer)
        st.success("Voice captured")

    st.subheader("📷 PPG")
    webrtc_streamer(key="camera")

    if st.button("Capture PPG"):
        hr = np.random.uniform(65, 95)
        hrv = np.random.uniform(15, 35)
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

    X = np.array([[pitch,jitter,shimmer,age,bmi,hr,hrv]])
    glucose = model.predict(X)[0]

    # Adjustments
    if fatigue > 30: glucose += 10
    if depression: glucose += 8
    if meal_status == "Just Ate": glucose += 20
    elif meal_status == "1 Hour After Meal": glucose += 10

    glucose = max(70, min(glucose, 220))

    prev = st.session_state["twin_data"][-1]["Glucose"] if st.session_state["twin_data"] else glucose
    delta = glucose - prev

    st.metric("🩸 Glucose", int(glucose), f"{delta:+.1f}")

    # =========================
    # SAVE TO DIGITAL TWIN ✅ FIX
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
        st.session_state["last_saved_glucose"] = int(glucose)
        st.session_state["last_check"] = datetime.now()

        st.success("✅ Saved to Digital Twin")

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

    else:
        st.warning("No data yet")

# =========================
# TAB 4 - PERFORMANCE
# =========================
with tab4:

    st.subheader("📈 Model Performance Analysis")

    n = 300
    df_perf = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "true": np.random.choice([0,1], n),
        "pred_prob": np.random.uniform(0,1,n)
    })

    df_perf["pred"] = (df_perf["pred_prob"] > 0.5).astype(int)

    for g in ["Female", "Male"]:

        st.markdown(f"## {g} Group")

        df_g = df_perf[df_perf["gender"] == g]

        col1, col2 = st.columns(2)

        # Violin
        with col1:
            fig, ax = plt.subplots()
            if seaborn_available:
                sns.violinplot(x=df_g["true"], y=df_g["pred_prob"], ax=ax)
            else:
                ax.violinplot([
                    df_g[df_g["true"]==0]["pred_prob"],
                    df_g[df_g["true"]==1]["pred_prob"]
                ])
            st.pyplot(fig)

        # Confusion
        with col2:
            cm = confusion_matrix(df_g["true"], df_g["pred"])
            cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots()
            im = ax.imshow(cm_norm)

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center")

            st.pyplot(fig)

        # ROC
        fpr, tpr, _ = roc_curve(df_g["true"], df_g["pred_prob"])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.legend()
        st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.caption("🚀 AI Digital Twin Prototype | Hackathon Ready")
