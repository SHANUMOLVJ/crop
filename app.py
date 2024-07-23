import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('crop_recommendation.csv')

# Define the columns to keep
columns_to_keep = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Crop']

# Check if all columns are present
missing_columns = [col for col in columns_to_keep if col not in data.columns]
if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
else:
    # Keep only the specified columns
    data = data[columns_to_keep]

    # Separate features and target
    X = data.drop('Crop', axis=1)
    y = data['Crop']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model to a file
    joblib.dump(clf, 'crop_recommendation_model.pkl')
    print("Model saved as crop_recommendation_model.pkl")

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
