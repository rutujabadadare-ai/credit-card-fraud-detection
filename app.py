import streamlit as st
import joblib, json
import numpy as np

# Load artifacts
model = joblib.load("fraud_model.pkl")
with open("feature_names.json") as f:
    feature_names = json.load(f)

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and predict if it's Fraudulent.")

# User inputs
inputs = []
for name in feature_names[:5]:   # just take first 5 features to keep it simple
    val = st.number_input(f"{name}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    X = np.array([inputs])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.write(f"Fraud Probability: **{prob:.2%}**")
    if pred == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
