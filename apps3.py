import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart Attack Analyser",
    page_icon="❤️",
    layout="centered"
)


model = joblib.load("models.pkl")

st.title("❤️ Heart Attack Prediction")

age = st.number_input("Age")
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.selectbox("CP(0-3)", [0,1,2,3])
chol = st.number_input("chol")
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("ca(0-3)", [0,1,2,3])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

if st.button("Predict"):
    features = np.array([[age, sex, cp,chol,oldpeak,
                          slope, ca, thal]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")
