import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Clinical No-Show Predictor", layout="centered")
st.title("🏥 Clinical Appointment No-Show Prediction")
st.markdown("Predict the likelihood of a patient missing their appointment.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join("models", "best_model.pkl")
        model = pickle.load(open(model_path, "rb"))
        return model
    except Exception as e:
        st.error("Model file not found. Please check deployment.")
        st.stop()

model = load_model()

st.markdown("---")

# -----------------------------
# Patient Details
# -----------------------------
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with col2:
    alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
    handicap = st.selectbox("Handicap", [0, 1, 2, 3, 4])
    sms_received = st.selectbox("SMS Reminder Sent", ["No", "Yes"])

st.markdown("---")

# -----------------------------
# Appointment Details
# -----------------------------
st.subheader("📅 Appointment Details")

scheduled_date = st.date_input("Scheduled Date")
appointment_date = st.date_input("Appointment Date")

if appointment_date < scheduled_date:
    st.error("Appointment date cannot be before scheduled date.")
    st.stop()

waiting_days = (appointment_date - scheduled_date).days

st.info(f"Calculated Waiting Days: {waiting_days}")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Encode categorical variables
    input_data = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Alcoholism": 1 if alcoholism == "Yes" else 0,
        "Handicap": handicap,
        "SMS_received": 1 if sms_received == "Yes" else 0,
        "waiting_days": waiting_days
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Patient Likely to MISS Appointment")
    else:
        st.success("✅ Patient Likely to ATTEND Appointment")

    st.metric("No-Show Probability", f"{probability * 100:.2f}%")

    st.markdown("---")

    # Business Recommendation
    if probability > 0.65:
        st.warning("Recommended Action: Send Reminder Call + SMS Follow-up.")
    elif probability > 0.45:
        st.info("Recommended Action: Send SMS Reminder.")
    else:
        st.success("Standard reminder protocol is sufficient.")

st.markdown("---")
st.caption("Tuned Random Forest Model | Streamlit Deployment | NST Sonipat")