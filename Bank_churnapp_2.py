# Imports
import pandas as pd
import streamlit as st
import joblib
import os

# Load pipeline
model_path = "churn_pipeline.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("MODEL LOADED SUCCESSFULLY")
else:
    st.warning(f"Model file {model_path} does not exist")

# App Heading
st.title("💳 Customer Churn Prediction")
st.write("Fill in the details below to predict the likelihood of churn:")

# Form Inputs
with st.form("Churn_Form"):
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Geography = st.selectbox("Geography", ["France", "Germany", "Italy", "Spain", "UK"])
    Tenure = st.number_input("Tenure (In months)", min_value=0)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0)
    PaymentMethod = st.selectbox("PaymentMethod", ["Bank transfer", "Credit card", "Electronic check", "Direct debit"])
    IsActiveMember = st.selectbox("IsActiveMember", ["Yes", "No"])

    Submitted = st.form_submit_button("Predict")

# Predict and display results
if Submitted:
    # Step 1: Create input DataFrame
    input_data = pd.DataFrame([{
        "Tenure": Tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Gender": Gender,
        "Geography": Geography,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "IsActiveMember": IsActiveMember
    }])

    # Step 2: Reorder columns to match pipeline
    input_data = input_data[pipeline.feature_names_in_]

    # Step 3: Prediction
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    # Step 4: Display result with colors
    if prediction == 1:
        st.error(f"⚠️ Prediction: Customer is likely to churn")
    else:
        st.success(f"✅ Prediction: Customer is not likely to churn")

    # Step 5: Probability Gauge
    st.subheader("Churn Probability")
    st.progress(int(probability*100))
    st.write(f"Probability: {probability*100:.2f}%")
