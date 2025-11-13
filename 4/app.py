import streamlit as st
import pandas as pd
import pickle

# --- Load model and encoders ---
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Remove target encoder if present
encoders.pop('Churn', None)

# --- Handle model and feature names ---
if isinstance(model_data, dict):
    loaded_model = model_data.get("model", model_data)
    feature_names = model_data.get("feature_names", None)
else:
    loaded_model = model_data
    feature_names = getattr(loaded_model, "feature_names_in_", None)

# --- App title ---
st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to predict whether they are likely to churn.")

# --- User input section ---
def user_input():
    data = {}
    data['gender'] = st.selectbox("Gender", ["Female", "Male"])
    data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
    data['Partner'] = st.selectbox("Partner", ["Yes", "No"])
    data['Dependents'] = st.selectbox("Dependents", ["Yes", "No"])
    data['tenure'] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    data['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"])
    data['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    data['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    data['OnlineSecurity'] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    data['OnlineBackup'] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    data['DeviceProtection'] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    data['TechSupport'] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    data['StreamingTV'] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    data['StreamingMovies'] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    data['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    data['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"])
    data['PaymentMethod'] = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    data['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, value=50.0)

    return pd.DataFrame([data])

# --- Collect user input ---
input_df = user_input()

# --- Encode categorical variables safely ---
for col, encoder in encoders.items():
    if col in input_df.columns:
        val = input_df[col].iloc[0]
        if val in encoder.classes_:
            input_df[col] = encoder.transform([val])
        else:
            # Assign default class if unseen
            input_df[col] = encoder.transform([encoder.classes_[0]])

# --- Ensure correct column order safely ---
if feature_names is not None:
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --- Prediction section ---
if st.button("Predict"):
    prediction = loaded_model.predict(input_df)
    pred_prob = loaded_model.predict_proba(input_df)

    result = "Churn" if prediction[0] == 1 else "No Churn"
    probability = pred_prob[0][1] * 100

    st.subheader("📈 Prediction Result")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Probability of Churn:** {probability:.2f}%")
