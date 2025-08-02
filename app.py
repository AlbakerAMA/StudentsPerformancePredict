import streamlit as st
import numpy as np
import joblib

# Load all model artifacts once when the app starts
@st.cache_resource
def load_model_artifacts():
    theta = joblib.load('manual_model.pkl')
    mean = joblib.load('mean.pkl')
    std = joblib.load('std.pkl')
    return theta, mean, std

theta, mean, std = load_model_artifacts()

st.title("Student Performance Predictor")
st.write("Fill in the student information to predict their math score.")

# Input fields with the EXACT same options and order as during training
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

def encode_features(gender, race, parent_edu, lunch, prep, reading, writing):
    """EXACTLY replicate the feature encoding used during training"""
    features = []
    
    # Gender (binary)
    features.append(1 if gender == "male" else 0)
    
    # Race (one-hot encoded, dropping first category)
    race_categories = ["group B", "group C", "group D", "group E"]  # group A is reference
    features += [1 if race == cat else 0 for cat in race_categories]
    
    # Parent education (one-hot encoded, dropping first category)
    edu_categories = [
        "high school", "some college", "associate's degree",
        "bachelor's degree", "master's degree"  # some high school is reference
    ]
    features += [1 if parent_edu == cat else 0 for cat in edu_categories]
    
    # Lunch (binary)
    features.append(1 if lunch == "standard" else 0)
    
    # Test prep (binary)
    features.append(1 if prep == "completed" else 0)
    
    # Numerical scores (not normalized yet)
    features.extend([reading, writing])
    
    return np.array(features)

def predict(features):
    """Make prediction matching the training process exactly"""
    features = features.reshape(1, -1)
    
    # Normalize ONLY the numerical features (last two)
    features[:, -2:] = (features[:, -2:] - mean[-2:]) / std[-2:]
    
    # Add bias term
    features = np.insert(features, 0, 1, axis=1)
    
    # Linear combination
    prediction = np.dot(features, theta)
    
    # Ensure reasonable score range
    return np.clip(prediction[0], 0, 100)

if st.button("Predict Math Score"):
    try:
        # Verify model artifacts
        st.write(f"Model coefficients shape: {theta.shape}")
        st.write(f"Mean values shape: {mean.shape}")
        st.write(f"Std values shape: {std.shape}")
        
        # Encode and predict
        features = encode_features(gender, race, parent_edu, lunch, prep, reading_score, writing_score)
        st.write(f"Encoded features: {features}")
        st.write(f"Feature count: {len(features)} (should match model coefficients - 1)")
        
        prediction = predict(features)
        st.success(f"Predicted Math Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
