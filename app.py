import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

# Define paths

model = RandomForestClassifier()
model_path = 'crop_recommendation_model.pkl'
data_path = ('Crop_recommendation.csv')
joblib.dump(model, 'crop_recommendation_model.pkl')
# Load the data
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.error(f"Data file {data_path} not found.")
    st.stop()

# Define the columns to keep
columns_to_keep = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Crop']

# Check if all columns are present
missing_columns = [col for col in columns_to_keep if col not in data.columns]
if missing_columns:
    st.error(f"Missing columns in the dataset: {missing_columns}")
    st.stop()
else:
    # Keep only the specified columns
    data = data[columns_to_keep]

    # Separate features and target
    X = data.drop('Crop', axis=1)
    y = data['Crop']

    # Define preprocessing for numeric features
    numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Combine preprocessing and model into a pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Create a pipeline with preprocessing and model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the trained pipeline to a file
    joblib.dump(model_pipeline, model_path)
    st.write(f"Model saved as {model_path}")

# Load the trained model
if os.path.exists(model_path):
    try:
        model_pipeline = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
else:
    st.error(f"Model file {model_path} not found.")
    st.stop()

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
    
    try:
        prediction = model_pipeline.predict(input_data)
        st.success(f"The recommended crop is: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
