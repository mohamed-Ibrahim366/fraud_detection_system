import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraud Detection System", page_icon="🚨", layout="centered")

# Load the preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('fraud_detection_model.pkl')

st.title("🚨 Fraud Detection in Transactions")
st.caption("AI-powered transaction analysis to detect potential frauds.")
st.markdown("---")

with st.form(key="fraud_form"):
    st.subheader("Enter Transaction Details:")

    amount = st.number_input("💵 Transaction Amount ($)", min_value=0.0, step=0.01)
    transaction_type = st.selectbox("Transaction Type", ["ATM Withdrawal", "Bill Payment", "POS Payment", "Bank Transfer", "Online Purchase"])
    time_of_transaction = st.slider("Time of Transaction (Hour)", 0, 23, format="%d")
    device_used = st.selectbox("Device Used", ["Mobile", "Tablet", "Desktop"])
    location = st.selectbox("Location", ["San Francisco", "New York", "Chicago", "Boston", "Houston", "Miami"])
    previous_fraud = st.slider("Number of Previous Frauds", 0, 5)
    account_age = st.slider("Account Age (Months)", 0, 120)
    number_of_transactions = st.slider("Number of Past Transactions (Last 24H)", 0, 50)
    payment_method = st.selectbox("Payment Method", ["Debit Card", "Credit Card", "UPI", "Net Banking"])

    submit_button = st.form_submit_button(label="Check for Fraud 🚀")

    if submit_button:
        with st.spinner('Analyzing transaction...'):
            # Create a DataFrame with the input data matching training columns
            input_data = pd.DataFrame([{
                'Transaction_Amount': amount,
                'Transaction_Type': transaction_type,
                'Time_of_Transaction': float(time_of_transaction),
                'Device_Used': device_used,
                'Location': location,
                'Previous_Fraudulent_Transactions': previous_fraud,
                'Account_Age': account_age,
                'Number_of_Transactions_Last_24H': number_of_transactions,
                'Payment_Method': payment_method
            }])

            # Apply the preprocessor to the input data
            input_features = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(input_features)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("⚠️ Warning: Fraudulent Transaction Detected!")
            else:
                st.success("✅ Transaction is Safe.")

st.markdown("---")
st.caption("Made with ❤️ by fighters Team")