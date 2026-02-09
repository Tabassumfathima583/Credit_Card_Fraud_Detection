# -----------------------------
# Ignore warnings
# -----------------------------
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# 1️⃣ Load trained model
# -----------------------------
model = joblib.load("xgb_creditcard_fraud_model.pkl")

# -----------------------------
# 2️⃣ Streamlit app title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to predict whether it is fraud or not.")

# -----------------------------
# 3️⃣ Input features
# -----------------------------
feature_names = [
    "Time", "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}:", value=0.0)

# -----------------------------
# 4️⃣ Scale Time & Amount
# -----------------------------
scaler = StandardScaler()
input_df = pd.DataFrame([input_data])
input_df[["Time","Amount"]] = scaler.fit_transform(input_df[["Time","Amount"]])

# -----------------------------
# 5️⃣ Make prediction
# -----------------------------
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Transaction Normal. Probability of Fraud: {probability:.2f}")

st.subheader("🔍 Model Explanation (Feature Importance)")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

st.bar_chart(importance_df.set_index("Feature"))

