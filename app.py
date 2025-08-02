import streamlit as st
import numpy as np
import joblib

st.title("Student Performance Predictor")
st.write("Fill in the student information to predict their math score.")

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

def encode_features(gender, race, parent_edu, lunch, prep, reading_score, writing_score):
    features = []

    # Gender: 0 = female, 1 = male
    features.append(1 if gender == 'male' else 0)

    # Race one-hot (drop 'group A')
    races = ['group B', 'group C', 'group D', 'group E']
    features += [1 if race == r else 0 for r in races]

    # Parental education one-hot (drop 'some high school')
    edu_levels = [
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
    features += [1 if parent_edu == level else 0 for level in edu_levels]

    # Lunch
    features.append(1 if lunch == 'standard' else 0)

    # Prep course
    features.append(1 if prep == 'completed' else 0)

    # Scores
    features.append(reading_score)
    features.append(writing_score)

    return np.array(features)

def predict(input_features):
    # Load model coefficients (theta), mean and std
    theta = joblib.load('manual_model.pkl')
    mean = joblib.load('mean.pkl')
    std = joblib.load('std.pkl')

    features = np.array(input_features).reshape(1, -1)
    
    # Verify feature dimensions
    if features.shape[1] != len(mean):
        raise ValueError(f"Mismatch in features vs mean/std shapes: features {features.shape[1]}, mean {len(mean)}")
    
    # Normalize features
    features = (features - mean) / std
    
    # Add bias term (intercept) and make prediction
    features_with_bias = np.insert(features, 0, 1, axis=1)  # Add column of 1s for bias term
    prediction = features_with_bias @ theta.T  # Matrix multiplication
    
    return float(prediction)

if st.button("Predict Math Score"):
    try:
        input_features = encode_features(gender, race, parent_edu, lunch, prep, reading_score, writing_score)
        prediction = predict(input_features)
        st.success(f"Predicted Math Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
