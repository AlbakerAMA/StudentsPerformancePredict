import streamlit as st
import numpy as np
import joblib

# Load model and normalization values
theta = joblib.load('manual_model.pkl')
mean = joblib.load('mean.pkl')
std = joblib.load('std.pkl')

# Ensure mean and std are flat arrays
mean = np.array(mean).flatten()
std = np.array(std).flatten()

# Define prediction function
def predict(features):
    features = np.array(features).reshape(1, -1)

    if features.shape[1] != mean.shape[0]:
        raise ValueError(f"Mismatch in features vs mean/std shapes: features {features.shape}, mean {mean.shape}, std {std.shape}")

    # Apply Z-score normalization to all features
    features = (features - mean) / std

    # Add bias term (1 for theta_0)
    features = np.insert(features, 0, 1, axis=1)

    return float(features @ theta)

# Streamlit UI
st.title("Student Performance Predictor")

# Input fields — adjust based on your trained model
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D"])
parent_edu = st.selectbox("Parental Level of Education", [
    "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.slider("Writing Score", min_value=0, max_value=100, step=1)

# Encode categorical features manually (must match training time encoding)
def encode_features(gender, race, parent_edu, lunch, prep):
    encoding = []

    # Example: 1 for male, 0 for female
    encoding.append(1 if gender == "male" else 0)

    # Race one-hot (example — must match your training encoding order)
    races = ["group A", "group B", "group C", "group D", "group E"]
    encoding += [1 if race == r else 0 for r in races]

    # Parental education one-hot
    educ_levels = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    encoding += [1 if parent_edu == level else 0 for level in educ_levels]

    # Lunch: 1 for standard, 0 for free/reduced
    encoding.append(1 if lunch == "standard" else 0)

    # Test prep: 1 if completed, 0 if none
    encoding.append(1 if prep == "completed" else 0)

    return encoding

# Collect inputs
encoded_inputs = encode_features(gender, race, parent_edu, lunch, prep)
numeric_inputs = [reading_score, writing_score]
input_features = encoded_inputs + numeric_inputs

# Predict button
if st.button("Predict Math Score"):
    try:
        prediction = predict(input_features)
        st.success(f"Predicted Math Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
