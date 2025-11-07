
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    artifacts = joblib.load('churn_prediction_model.pkl')
    return artifacts

# Load model artifacts
try:
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model file not found. Please ensure 'churn_prediction_model.pkl' is in the same directory.")
    st.stop()

# App title and description
st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown("""
Predict which customers are at risk of leaving and take proactive retention actions.
* **Model**: Random Forest Classifier
* **Accuracy**: 86%
* **Churn Detection Rate**: 45%
""")

# Sidebar for customer input
st.sidebar.header("📝 Customer Information")

# Create input fields based on feature names
def create_inputs():
    inputs = {}
    
    # Numerical inputs
    inputs['CreditScore'] = st.sidebar.slider("Credit Score", 300, 850, 650)
    inputs['Age'] = st.sidebar.slider("Age", 18, 80, 40)
    inputs['Tenure'] = st.sidebar.slider("Tenure (years)", 0, 10, 5)
    inputs['Balance'] = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
    inputs['NumOfProducts'] = st.sidebar.slider("Number of Products", 1, 4, 1)
    inputs['EstimatedSalary'] = st.sidebar.number_input("Estimated Salary", 0.0, 300000.0, 75000.0)
    
    # Categorical inputs
    inputs['Gender'] = st.sidebar.selectbox("Gender", ["Female", "Male"])
    inputs['IsActiveMember'] = st.sidebar.selectbox("Active Member", ["Yes", "No"])
    inputs['HasCrCard'] = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
    
    # Geography - one hot encoded
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    inputs['Geo_France'] = 1 if geography == "France" else 0
    inputs['Geo_Germany'] = 1 if geography == "Germany" else 0
    inputs['Geo_Spain'] = 1 if geography == "Spain" else 0
    
    # Encode categorical variables
    inputs['Gender'] = 0 if inputs['Gender'] == "Female" else 1
    inputs['IsActiveMember'] = 1 if inputs['IsActiveMember'] == "Yes" else 0
    inputs['HasCrCard'] = 1 if inputs['HasCrCard'] == "Yes" else 0
    
    return inputs

# Get user inputs
customer_data = create_inputs()

# Create dataframe in correct feature order
input_df = pd.DataFrame([customer_data])[feature_names]

# Prediction section
st.header("🎯 Churn Prediction Results")

if st.button("Predict Churn Probability", type="primary"):
    try:
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")
            
        with col2:
            if prediction == 1:
                st.error("🚨 High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")
                
        with col3:
            if probability > 0.7:
                st.warning("🔴 Immediate Action Needed")
            elif probability > 0.4:
                st.info("🟡 Monitor Closely")
            else:
                st.success("🟢 Low Risk")
        
        # Risk analysis
        st.subheader("📈 Risk Analysis")
        if prediction == 1:
            st.error("**Recommended Actions:**")
            st.write("- Contact customer for feedback")
            st.write("- Offer personalized retention offer") 
            st.write("- Assign to dedicated account manager")
            st.write("- Schedule follow-up in 1 week")
        else:
            st.success("**Customer appears stable:**")
            st.write("- Continue regular engagement")
            st.write("- Monitor for any changes in behavior")
            st.write("- Consider cross-selling opportunities")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Model information
with st.expander("ℹ️ Model Information"):
    st.write("**Model Performance:**")
    st.write("- Overall Accuracy: 86%")
    st.write("- Churn Detection Rate: 45%")
    st.write("- Precision on Churn Predictions: 77%")
    st.write("")
    st.write("**Top Factors Influencing Churn:**")
    st.write("1. Customer Age")
    st.write("2. Account Balance")  
    st.write("3. Geography (Germany highest risk)")
    st.write("4. Active Member Status")
    st.write("")
    st.write("**Limitations:**")
    st.write("- Model misses 55% of actual churn cases")
    st.write("- Use predictions as guidance, not absolute truth")
    st.write("- Combine with business intuition and customer feedback")

# Batch prediction section
with st.expander("📁 Batch Prediction (Upload CSV)"):
    uploaded_file = st.file_uploader("Upload customer data CSV", type="csv")
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_data)} customers")
            
            if st.button("Predict Batch"):
                st.info("Batch prediction feature would be implemented here")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: Random Forest | Data: Customer Banking Information")
