import streamlit as st
import numpy as np
import joblib

# Load model and normalization values
try:
    theta = joblib.load('manual_model.pkl')
    mean = joblib.load('mean.pkl')
    std = joblib.load('std.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

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
    features = np.array(features, dtype=float).reshape(1, -1)
    
    # Normalize features using the saved mean and std
    # Check if mean and std are arrays (normalized all features) or scalars (normalized only numerical)
    if hasattr(mean, 'shape') and len(mean.shape) > 0 and mean.shape[0] == features.shape[1]:
        # Mean and std were calculated for all features
        features = (features - mean) / std
    elif hasattr(mean, 'shape') and len(mean.shape) > 0 and mean.shape[0] == 2:
        # Mean and std were calculated only for the last 2 features (reading, writing)
        features[:, -2:] = (features[:, -2:] - mean) / std
    else:
        # Mean and std are scalars - apply to last 2 features
        features[:, -2:] = (features[:, -2:] - mean) / std
    
    # Add bias term (intercept) at the beginning  
    features_with_bias = np.insert(features, 0, 1, axis=1)
    
    # Make prediction
    prediction = float(features_with_bias @ theta)
    return prediction

# Streamlit UI
st.title("📊 Math Score Predictor")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", gender_options)
        race = st.selectbox("Race/Ethnicity", race_options)
        parent_edu = st.selectbox("Parental Level of Education", parent_edu_options)
    
    with col2:
        lunch = st.selectbox("Lunch Type", lunch_options)
        prep_course = st.selectbox("Test Preparation Course", prep_options)
    
    st.subheader("Test Scores")
    col3, col4 = st.columns(2)
    with col3:
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50, step=1)
    with col4:
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50, step=1)
    
    # Prediction button
    submitted = st.form_submit_button("Predict Math Score", type="primary")
    
    if submitted:
        try:
            # Assemble all features in the correct order
            cat_features = (
                one_hot_encode(gender, gender_options) +
                one_hot_encode(race, race_options) +
                one_hot_encode(parent_edu, parent_edu_options) +
                one_hot_encode(lunch, lunch_options) +
                one_hot_encode(prep_course, prep_options)
            )
            
            # Add numerical features
            num_features = [reading_score, writing_score]
            input_features = cat_features + num_features
            
            # Make prediction
            prediction = predict(input_features)
            
            # Display result
            st.success(f"**Predicted Math Score: {prediction:.1f}**")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please ensure all model files (manual_model.pkl, mean.pkl, std.pkl) are in the correct format.")
