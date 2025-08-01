import streamlit as st
import numpy as np
import joblib

# Load model and normalization values
theta = joblib.load('manual_model.pkl')
mean = joblib.load('mean.pkl')
std = joblib.load('std.pkl')

# Define categorical encodings (must match training phase)
gender_options = ['female', 'male']
race_options = ['group A', 'group B', 'group C', 'group D', 'group E']
parent_edu_options = [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
]
lunch_options = ['standard', 'free/reduced']
prep_options = ['none', 'completed']

# One-hot encoding function
def one_hot_encode(value, options):
    return [1 if value == option else 0 for option in options]

# Define prediction function
def predict(features):
    features = np.array(features).reshape(1, -1)

    # Convert mean and std to flat NumPy arrays if not already
    global mean, std
    mean = np.array(mean).flatten()
    std = np.array(std).flatten()

    # Make sure mean and std only apply to the last two features
    numeric = features[:, -2:].astype(float)

    if mean.shape[0] != 2 or std.shape[0] != 2:
        raise ValueError(f"Expected mean and std of shape (2,), got mean {mean.shape}, std {std.shape}")

    # Normalize
    normed = (numeric - mean) / std
    features[:, -2:] = normed

    # Add bias term (1 for theta_0)
    features = np.insert(features, 0, 1, axis=1)
    return float(features @ theta)

# Streamlit UI
st.title("Predict Your Math Score")

# Inputs
gender = st.selectbox("Gender", gender_options)
race = st.selectbox("Race/Ethnicity", race_options)
parent_edu = st.selectbox("Parental Level of Education", parent_edu_options)
lunch = st.selectbox("Lunch Type", lunch_options)
prep_course = st.selectbox("Test Preparation Course", prep_options)

reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

# Assemble final feature vector
if st.button("Predict Math Score"):
    cat_features = (
        one_hot_encode(gender, gender_options) +
        one_hot_encode(race, race_options) +
        one_hot_encode(parent_edu, parent_edu_options) +
        one_hot_encode(lunch, lunch_options) +
        one_hot_encode(prep_course, prep_options)
    )
    num_features = [reading_score, writing_score]
    input_features = cat_features + num_features

    prediction = predict(input_features)
    st.success(f"Predicted Math Score: {prediction:.2f}")
