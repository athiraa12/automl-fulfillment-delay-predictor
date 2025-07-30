import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
model = load("app.py/tpot_lightgbm_model.pkl")


# Streamlit app config
st.set_page_config(page_title="📦 Warehouse Delay Predictor", layout="centered")

# App title and intro
st.title("📦 Warehouse Delay Predictor")
st.markdown("Predict warehouse delays and get actionable recommendations based on today's inputs.")

# Input form
st.subheader("🔧 Input Warehouse Conditions")

day = st.selectbox("Day of the Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
orders_received = st.slider("Orders Received Today", 50, 1000, 300)
workers_available = st.slider("Workers Available", 5, 50, 15)
avg_pick_time = st.slider("Average Pick Time (minutes)", 2.0, 10.0, 5.0)
shift_hours = st.slider("Shift Length (hours)", 6, 12, 8)

# Encode day to numeric
day_code = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(day)

# Prepare input DataFrame
input_df = pd.DataFrame([[day_code, orders_received, workers_available, avg_pick_time, shift_hours]],
                        columns=["day", "orders_received", "workers_available", "avg_pick_time", "shift_hours"])

# Run prediction
if st.button("🔍 Predict Delay"):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error(f"🚨 Delay Likely! (Confidence: {prediction_prob:.2%})")
    else:
        st.success(f"✅ On Time! (Confidence: {1 - prediction_prob:.2%})")

    # Recommendation logic
    st.subheader("🧠 Smart Recommendations")
    workload_ratio = orders_received / workers_available

    if workload_ratio > 20 and avg_pick_time > 6:
        st.info("📌 Add more workers and reorganize shelves to reduce pick time.")
    elif avg_pick_time > 7:
        st.info("📦 Reconfigure item layout or increase automation to reduce pick time.")
    elif workload_ratio > 25:
        st.info("👥 Schedule backup staff during peak order periods.")
    elif shift_hours < 7:
        st.info("⏱️ Extend shift hours or overlap with a second shift.")
    else:
        st.info("✅ Operations look efficient based on today's inputs.")

# Add a divider line
st.markdown("---")

# Footer message
st.markdown(
    "<div style='text-align: center; font-size: 16px;'>"
    "🧠 <em>This tool can help logistics teams prioritize at-risk orders and balance warehouse loads.</em><br>"
    "</div>",
    unsafe_allow_html=True
)
