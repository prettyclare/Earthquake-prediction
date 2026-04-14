import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Earthquake Significance Predictor",
    page_icon="🌍",
    layout="centered"
)

# ── Load model artifacts ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ── Header ─────────────────────────────────────────────────────
st.title("🌍 Earthquake Significance Predictor")
st.markdown(
    """
    This app uses a machine learning model trained on over **1 million USGS seismic 
    records (1900–2026)** to predict whether a seismic event is likely to be 
    **significant** — defined as magnitude ≥ 5.0.
    
    Enter the event parameters below and click **Predict**.
    """
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this model")
    st.markdown(
        """
        **Model:** XGBoost Classifier  
        **Task:** Binary classification  
        **Target:** Magnitude ≥ 5.0 = Significant  
        **Training data:** USGS Global Earthquake Dataset  
        **Features used:**
        - Focal depth (km)
        - Geographic coordinates
        - Depth category (shallow / intermediate / deep)
        - Ring of Fire location flag
        - Detection network quality metrics
        
        ---
        **COM763 Advanced Machine Learning**  
        Portfolio Task 1
        """
    )

# ── Input form ─────────────────────────────────────────────────
st.subheader("Event Parameters")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude (°)",
        min_value=-90.0, max_value=90.0,
        value=35.0, step=0.1,
        help="Positive = North, Negative = South"
    )
    longitude = st.number_input(
        "Longitude (°)",
        min_value=-180.0, max_value=180.0,
        value=139.0, step=0.1,
        help="Positive = East, Negative = West"
    )
    depth = st.slider(
        "Focal Depth (km)",
        min_value=0, max_value=750,
        value=30,
        help="Shallow < 70km  |  Intermediate 70–300km  |  Deep > 300km"
    )

with col2:
    nst = st.number_input(
        "No. of seismic stations (nst)",
        min_value=0, max_value=1000,
        value=20,
        help="Number of stations that detected the event"
    ) if "nst" in feature_cols else 20

    gap = st.number_input(
        "Azimuthal gap (°)",
        min_value=0.0, max_value=360.0,
        value=80.0, step=1.0,
        help="Largest azimuthal gap between stations"
    ) if "gap" in feature_cols else 80.0

    rms = st.number_input(
        "RMS amplitude",
        min_value=0.0, max_value=10.0,
        value=0.5, step=0.01,
        help="Root mean square of travel time residuals"
    ) if "rms" in feature_cols else 0.5

    dmin = st.number_input(
        "Minimum station distance (dmin)",
        min_value=0.0, max_value=10.0,
        value=0.1, step=0.01,
        help="Horizontal distance to nearest station (degrees)"
    ) if "dmin" in feature_cols else 0.1

# ── Info labels ────────────────────────────────────────────────
if depth < 70:
    depth_label = "🟡 Shallow (< 70 km) — highest surface impact"
elif depth < 300:
    depth_label = "🟠 Intermediate (70–300 km)"
else:
    depth_label = "🔵 Deep (> 300 km) — reduced surface impact"
st.caption(f"Depth classification: {depth_label}")

in_rof = int(abs(latitude) < 60 and (longitude < -60 or longitude > 100))
st.caption(f"Location: {'✅ Within Ring of Fire' if in_rof else '❌ Outside Ring of Fire'}")

st.divider()

# ── Feature builder ────────────────────────────────────────────
def build_input(lat, lon, dep, nst_v, gap_v, rms_v, dmin_v):
    row = {
        "depth":           dep,
        "latitude":        lat,
        "longitude":       lon,
        "abs_latitude":    abs(lat),
        "depth_category":  0 if dep < 70 else (1 if dep < 300 else 2),
        "in_ring_of_fire": int(abs(lat) < 60 and (lon < -60 or lon > 100)),
        "nst":             nst_v,
        "gap":             gap_v,
        "rms":             rms_v,
        "dmin":            dmin_v,
        "horizontalError": 1.0,
        "depthError":      1.5,
        "magNst":          10,
        "magType_encoded": 0,
        "year":            2024,
        "month":           6,
        "hour":            12,
    }
    input_df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_cols]

# ── Predict ────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary", use_container_width=True):

    X_input = build_input(latitude, longitude, depth, nst, gap, rms, dmin)
    pred    = model.predict(X_input)[0]
    proba   = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(
            f"⚠️ **SIGNIFICANT** event predicted\n\n"
            f"This event is predicted to be magnitude ≥ 5.0"
        )
    else:
        st.success(
            f"✅ **Not significant** event predicted\n\n"
            f"This event is predicted to be magnitude < 5.0"
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Significance probability",     f"{proba:.3f}")
    m2.metric("Non-significance probability", f"{1 - proba:.3f}")
    m3.metric("Predicted class", "Significant" if pred == 1 else "Not significant")

    st.markdown("**Confidence breakdown**")
    st.progress(float(proba), text=f"Significance confidence: {proba:.1%}")

    st.divider()

    with st.expander("See input features sent to model"):
        st.dataframe(
            build_input(latitude, longitude, depth, nst, gap, rms, dmin).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    st.caption(
        "⚠️ This tool is for educational purposes only. "
        "It is not a substitute for professional seismic monitoring systems."
    )