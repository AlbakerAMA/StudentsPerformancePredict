import streamlit as st
import numpy as np
import joblib

# Load model and normalization values
theta = joblib.load('manual_model.pkl')
mean = joblib.load('mean.pkl')
std = joblib.load('std.pkl')

# Define prediction function
def predict(features):
    features = np.array(features).reshape(1, -1)
    # Apply Z-score normalization
    features = (features - mean) / std
    # Add bias term (1 for theta_0)
    features = np.insert(features, 0, 1, axis=1)
    return float(features @ theta)

# Streamlit UI
st.title("Student Performance Predictor")

# Input fields (example with 3 features)
study_time = st.number_input("Study Time (hours per week)", min_value=0.0, max_value=20.0, step=0.5)
failures = st.number_input("Number of Past Class Failures", min_value=0, max_value=4, step=1)
absences = st.number_input("Number of Absences", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Predict Performance"):
    input_data = [study_time, failures, absences]
    prediction = predict(input_data)
    st.success(f"Predicted Final Grade: {prediction:.2f}")
