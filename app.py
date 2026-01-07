import streamlit as st
import numpy as np
import joblib

model = joblib.load("mobile_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Mobile Price Predictor", layout="centered")

st.title("📱 Mobile Price Range Predictor")
st.write("Enter basic mobile specifications")

st.divider()

# ✅ ONLY 4 INPUTS (MATCH MODEL)
memory = st.number_input("RAM (GB)", 1, 32, 8)
storage = st.number_input("Storage (GB)", 8, 512, 128)
rating = st.slider("User Rating", 1.0, 5.0, 4.2)
original_price = st.number_input("Original Price (₹)", 5000, 150000, 30000)

st.divider()

# ✅ BUTTON ONLY DOES PREDICTION
if st.button("🔮 Predict Price Range"):
    input_data = np.array([[memory, storage, rating, original_price]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    price_map = {
        0: "💰 Budget",
        1: "💵 Mid Range",
        2: "💎 Premium",
        3: "👑 Flagship"
    }

    st.success(f"Predicted Price Range: **{price_map[prediction]}**")
