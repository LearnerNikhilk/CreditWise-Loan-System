import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the trained model and scaler
# Make sure 'loan_model.pkl' and 'scaler.pkl' are in the same folder as this file!
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('loan_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

# 2. App Styling & Header
st.set_page_config(page_title="CreditWise Loan Predictor", page_icon="🏦")
st.title("🏦 CreditWise Loan Eligibility Predictor")
st.markdown("Enter the applicant's details below to check if the loan is likely to be **Approved**.")
st.divider()

# 3. Input Form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
        co_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
        score = st.slider("Credit Score", 300, 850, 700)
        loans = st.number_input("Number of Existing Loans", 0, 10, 0)
        
    with col2:
        dti = st.number_input("DTI Ratio (e.g. 0.35)", 0.0, 1.0, 0.3, step=0.01)
        savings = st.number_input("Total Savings ($)", min_value=0, value=1000)
        collateral = st.number_input("Collateral Value ($)", min_value=0, value=0)
        
        # Categorical selections (Adjusted for Label Encoding)
        edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        prop = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit = st.form_submit_button("Predict Loan Status")

# 4. Prediction Logic
if submit:
    if model is not None and scaler is not None:
        try:
            # Map categories to the numbers your model expects
            edu_val = 1 if edu == "Graduate" else 0
            prop_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
            
            # Create the input array in the EXACT order your model was trained on
            raw_input = np.array([[
                income, co_income, score, loans, dti, savings, collateral, edu_val, prop_map[prop]
            ]])
            
            # Apply Scaling
            scaled_input = scaler.transform(raw_input)
            
            # Make Prediction
            prediction = model.predict(scaled_input)
            probability = model.predict_proba(scaled_input)[0][1] * 100
            
            st.divider()
            if prediction[0] == 1 or prediction[0] == "Yes":
                st.success(f"🎉 **Loan Approved!**")
                st.balloons()
            else:
                st.error(f"❌ **Loan Rejected.**")
            
            st.info(f"**Approval Confidence:** {probability:.2f}%")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Model or Scaler not found. Please upload 'loan_model.pkl' and 'scaler.pkl' to this folder.")
