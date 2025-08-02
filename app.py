import streamlit as st
import numpy as np
import joblib

# Load model and normalization values
try:
    theta = joblib.load('manual_model.pkl')
    mean = joblib.load('mean.pkl')
    std = joblib.load('std.pkl')
    
    st.write(f"Model loaded: theta shape = {theta.shape}")
    st.write(f"Mean values: {mean}")
    st.write(f"Std values: {std}")
    
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
    
    # Debug: Show raw features
    st.write(f"Raw features shape: {features.shape}")
    st.write(f"Raw features: {features}")
    
    # Separate categorical and numerical features
    # Assuming last 2 features are reading and writing scores
    categorical_features = features[:, :-2]
    numerical_features = features[:, -2:]
    
    st.write(f"Categorical features: {categorical_features}")
    st.write(f"Numerical features before normalization: {numerical_features}")
    
    # Normalize only the numerical features (reading, writing scores)
    if std is not None and mean is not None:
        # Ensure mean and std are arrays with correct shape
        if isinstance(mean, (int, float)):
            mean_array = np.array([mean, mean])
        else:
            mean_array = np.array(mean)
            
        if isinstance(std, (int, float)):
            std_array = np.array([std, std])
        else:
            std_array = np.array(std)
            
        # Avoid division by zero
        std_array = np.where(std_array == 0, 1, std_array)
        
        normalized_numerical = (numerical_features - mean_array) / std_array
        st.write(f"Normalized numerical features: {normalized_numerical}")
    else:
        normalized_numerical = numerical_features
        st.warning("No normalization applied - mean/std not loaded properly")
    
    # Combine features back
    final_features = np.concatenate([categorical_features, normalized_numerical], axis=1)
    
    # Add bias term (intercept)
    final_features = np.insert(final_features, 0, 1, axis=1)
    
    st.write(f"Final features shape: {final_features.shape}")
    st.write(f"Final features: {final_features}")
    st.write(f"Theta shape: {theta.shape}")
    
    # Make prediction
    prediction = float(final_features @ theta)
    
    return prediction

# Streamlit UI
st.title("🎯 Math Score Predictor")
st.write("Enter student information to predict their math score")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Demographics")
    gender = st.selectbox("Gender", gender_options)
    race = st.selectbox("Race/Ethnicity", race_options)
    parent_edu = st.selectbox("Parental Level of Education", parent_edu_options)

with col2:
    st.subheader("Additional Information")
    lunch = st.selectbox("Lunch Type", lunch_options)
    prep_course = st.selectbox("Test Preparation Course", prep_options)

st.subheader("Test Scores")
col3, col4 = st.columns(2)
with col3:
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50, step=1)
with col4:
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50, step=1)

# Show feature encoding for debugging
with st.expander("🔍 Debug Information"):
    st.write("**Feature Encoding Preview:**")
    
    gender_encoded = one_hot_encode(gender, gender_options)
    race_encoded = one_hot_encode(race, race_options)
    parent_edu_encoded = one_hot_encode(parent_edu, parent_edu_options)
    lunch_encoded = one_hot_encode(lunch, lunch_options)
    prep_encoded = one_hot_encode(prep_course, prep_options)
    
    st.write(f"Gender ({gender}): {gender_encoded}")
    st.write(f"Race ({race}): {race_encoded}")
    st.write(f"Parent Education ({parent_edu}): {parent_edu_encoded}")
    st.write(f"Lunch ({lunch}): {lunch_encoded}")
    st.write(f"Prep Course ({prep_course}): {prep_encoded}")
    
    total_features = len(gender_encoded) + len(race_encoded) + len(parent_edu_encoded) + len(lunch_encoded) + len(prep_encoded) + 2
    st.write(f"Total feature count: {total_features}")

# Prediction button
if st.button("🎯 Predict Math Score", type="primary"):
    try:
        # Assemble categorical features
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
        
        st.write(f"Input features length: {len(input_features)}")
        
        # Make prediction
        prediction = predict(input_features)
        
        # Display result
        st.success(f"🎯 **Predicted Math Score: {prediction:.2f}**")
        
        # Add some context
        if prediction >= 80:
            st.balloons()
            st.write("🌟 Excellent performance expected!")
        elif prediction >= 60:
            st.write("👍 Good performance expected!")
        elif prediction >= 40:
            st.write("📚 Room for improvement - consider additional study!")
        else:
            st.write("💪 Significant study and support recommended!")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please check that your model files are compatible with the input format.")

# Add information about the model
st.sidebar.header("ℹ️ About This Model")
st.sidebar.write("""
This linear regression model predicts math scores based on:
- Student demographics
- Parental education level
- Lunch program participation
- Test preparation completion
- Reading and writing scores

The model uses one-hot encoding for categorical variables and normalization for numerical scores.
""")

st.sidebar.header("📁 Required Files")
st.sidebar.write("""
Make sure these files are in the same directory:
- `manual_model.pkl` - Trained model coefficients
- `mean.pkl` - Mean values for normalization
- `std.pkl` - Standard deviation for normalization
""")
