import streamlit as st
import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('crop_recommendation_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure that the model has been trained and saved as 'crop_recommendation_model.pkl'")

# Streamlit app interface
st.title("Crop Recommendation System")

# Input features for prediction
nitrogen = st.number_input("Nitrogen", min_value=0)
phosphorus = st.number_input("Phosphorus", min_value=0)
potassium = st.number_input("Potassium", min_value=0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph_value = st.number_input("pH Value", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Predict crop recommendation
if st.button("Predict Crop"):
    input_data = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'pH_Value': [ph_value],
        'Rainfall': [rainfall]
    })

    prediction = model.predict(input_data)
    st.success(f"The recommended crop is: {prediction[0]}")
