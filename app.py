import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

def predict_diabetes(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return "Diabetic" if prediction[0] > 0.5 else "Non-Diabetic"

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")

image = Image.open("img.webp")
st.image(image, use_container_width=True)

st.subheader("Enter the following details:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=6)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=148)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=35)
insulin = st.number_input("Insulin", min_value=0, max_value=500, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=33.6)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input("Age", min_value=18, max_value=120, value=50)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

if st.button("Predict Diabetes"):
    result = predict_diabetes(input_data)
    if result == "Diabetic":
        st.markdown(f"<h3 style='color: red; font-size: 24px;'>The prediction result is: <b>{result}</b></h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; font-size: 24px;'>The prediction result is: <b>{result}</b></h3>", unsafe_allow_html=True)

st.markdown("""
### Factors That Contribute to Diabetes Risk:

The model uses the following factors to predict the likelihood of diabetes:

- **Pregnancies**: A higher number of pregnancies may be linked to gestational diabetes.
- **Glucose**: High blood sugar levels indicate insulin resistance.
- **Blood Pressure**: Hypertension is associated with increased diabetes risk.
- **Skin Thickness**: Higher values may indicate fat accumulation and insulin resistance.
- **Insulin**: Elevated insulin levels suggest insulin resistance.
- **BMI (Body Mass Index)**: Obesity is a leading cause of diabetes.
- **Diabetes Pedigree Function**: A higher value suggests a genetic predisposition.
- **Age**: Older age increases diabetes risk due to metabolic changes.

Maintaining a healthy lifestyle, managing weight, and regular checkups can help mitigate these risks.
""")

st.markdown(
    "<h3 style='color: blue; text-align: right;'>Made with ❤️ by <span style='color: red;'>Aiswarya</span></h3>", 
    unsafe_allow_html=True
)
