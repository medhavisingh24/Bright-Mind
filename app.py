import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("autism_model.pkl")

st.title("Autism Prediction App")
st.write("Enter the details below to predict autism likelihood")

# Example inputs
age = st.number_input("Age", min_value=0, max_value=100, value=18)
gender = st.radio("Gender", ["Male", "Female"])
screening_score = st.slider("Screening Score", 0, 10, 5)

# Convert inputs for model
gender_num = 1 if gender == "Male" else 0
features = np.array([[age, gender_num, screening_score]])

if st.button("Predict"):
    prediction = model.predict(features)
    result = "Autism Likely" if prediction[0] == 1 else "No Autism"
    st.success(result)
