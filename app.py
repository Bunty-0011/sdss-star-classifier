"""
================================================================================
STREAMLIT APP — Stellar Object Classifier (GALAXY / QSO / STAR)
================================================================================
Ye app `model.pkl` aur `label_encoder.pkl` ko load karta hai (jo
train_and_save_model.py se generate hote hain) aur user se photometric
values lekar prediction deta hai.

Run locally: streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Stellar Object Classifier", page_icon="🌌", layout="centered")

st.title("🌌 Stellar Object Classifier")
st.write(
    "Predicts whether an SDSS sky object is a **GALAXY**, **QSO** (quasar), or **STAR** "
    "based on photometric and spectral measurements. Model: LightGBM, test accuracy ≈ 0.968."
)

# ---------------------------------------------------------------
# Load model (cached so it only loads once, not on every interaction)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

try:
    model, le = load_model()
except FileNotFoundError:
    st.error(
        "model.pkl / label_encoder.pkl not found. Run `python train_and_save_model.py` "
        "first, and make sure both files are in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.subheader("Enter object measurements")

col1, col2 = st.columns(2)

with col1:
    alpha = st.number_input("alpha (Right Ascension)", value=180.0, format="%.6f")
    delta = st.number_input("delta (Declination)", value=0.0, format="%.6f")
    u = st.number_input("u (ultraviolet band)", value=20.0, format="%.4f")
    g = st.number_input("g (green band)", value=19.0, format="%.4f")
    r = st.number_input("r (red band)", value=18.5, format="%.4f")

with col2:
    i = st.number_input("i (infrared band)", value=18.0, format="%.4f")
    z = st.number_input("z (infrared band)", value=17.8, format="%.4f")
    redshift = st.number_input("redshift", value=0.05, format="%.6f")
    spectral_type = st.selectbox(
        "spectral_type",
        options=["STAR_RED_DWARF", "STAR_WHITE_DWARF", "GALAXY", "QSO", "OTHER"],
        # NOTE: replace this list with the ACTUAL unique values from your
        # df['spectral_type'].unique() in the training data — this is a placeholder.
    )

predict_btn = st.button("Predict class", type="primary")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if predict_btn:
    # Build a single-row dataframe matching the training feature set.
    # Feature engineering here MUST exactly match train_and_save_model.py.
    row = pd.DataFrame([{
        "alpha": alpha,
        "delta": delta,
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "redshift": redshift,
        "spectral_type": spectral_type,
    }])

    row["u_g"] = row["u"] - row["g"]
    row["g_r"] = row["g"] - row["r"]
    row["r_i"] = row["r"] - row["i"]
    row["i_z"] = row["i"] - row["z"]
    row["ug"] = row["u"] * row["g"]
    row["gr"] = row["g"] * row["r"]
    row["ri"] = row["r"] * row["i"]
    row["iz"] = row["i"] * row["z"]

    pred_encoded = model.predict(row)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(row)[0]
    proba_df = pd.DataFrame({"class": le.classes_, "probability": proba}).sort_values(
        "probability", ascending=False
    )

    st.success(f"### Predicted class: **{pred_label}**")
    st.write("Class probabilities:")
    st.dataframe(proba_df, hide_index=True, use_container_width=True)
    st.bar_chart(proba_df.set_index("class"))

st.divider()
st.caption(
    "Model trained on SDSS stellar classification data. "
    "See the accompanying notebook for full EDA, feature engineering, and evaluation details."
)