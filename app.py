import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image



# Load your trained model and scaler
model = load_model('trained_model.h5')  # Replace with your actual model path
scaler = joblib.load('scaler.pkl')  # Replace with your actual scaler file

# Function for making predictions
def predict_diabetes(data):
    # Scale the input data
    data_scaled = scaler.transform(data)
    # Make prediction
    prediction = model.predict(data_scaled)
    return "Diabetic" if prediction[0] > 0.5 else "Non-Diabetic"

# Set up Streamlit page
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Title of the web app
st.title("Diabetes Prediction App")

# Display an image (replace with your image file)
image = Image.open("img.webp")  # Replace with the path to your image
st.image(image, use_container_width=True)

# Input fields for prediction with default values
st.subheader("Enter the following details:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=6)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=148)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=35)
insulin = st.number_input("Insulin", min_value=0, max_value=500, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=33.6)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input("Age", min_value=18, max_value=120, value=50)

# Prepare the input data for prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Display prediction button
# Display prediction result with styling
if st.button("Predict Diabetes"):
    result = predict_diabetes(input_data)
    
    # Highlight result based on prediction
    if result == "Diabetic":
        st.markdown(f"<h3 style='color: red; font-size: 24px;'>The prediction result is: <b>{result}</b></h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; font-size: 24px;'>The prediction result is: <b>{result}</b></h3>", unsafe_allow_html=True)


# Add an explanation text
# Add an explanation text
st.markdown("""
### Factors That Contribute to Diabetes Risk:

The model uses the following factors to predict the likelihood of diabetes. Each of these factors is associated with an increased risk of developing type 2 diabetes:

- **Pregnancies**: A higher number of pregnancies may be linked to gestational diabetes or a higher likelihood of developing type 2 diabetes later in life.
  
- **Glucose**: High blood sugar (glucose) levels can indicate insulin resistance or poor blood sugar control, which is a major risk factor for diabetes.

- **Blood Pressure**: Hypertension (high blood pressure) is often associated with insulin resistance and increased risk for type 2 diabetes.

- **Skin Thickness**: High skin thickness may be related to increased fat accumulation, which can indicate insulin resistance and a higher risk of diabetes.

- **Insulin**: Elevated insulin levels in the blood can indicate insulin resistance, a common precursor to type 2 diabetes.

- **BMI (Body Mass Index)**: Higher BMI values are strongly correlated with obesity, which is one of the leading causes of type 2 diabetes. Excess body fat can lead to insulin resistance.

- **Diabetes Pedigree Function**: A higher value suggests a genetic predisposition to diabetes. If your family history shows diabetes, your risk may be higher.

- **Age**: Age is a significant factor. As people get older, the risk of developing type 2 diabetes increases, partly due to changes in metabolism and decreased physical activity.

These factors, when combined, help the model assess the likelihood of a person developing diabetes. Maintaining a healthy lifestyle, managing weight, and regular health checkups can help mitigate the risks associated with these factors.
""")


# Footer

st.markdown(
    "<h3 style='color: blue; text-align: right;'>Made with ❤️ by <span style='color: red;'>Aiswarya</span></h3>", 
    unsafe_allow_html=True
)
