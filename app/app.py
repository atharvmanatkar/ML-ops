import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
from src.predict import predict
from langchain.explain import explain_prediction

st.title("Anomaly Detection System")

# Load dataset to get feature names
df = pd.read_csv("data/raw/Ai-data.csv")
feature_columns = df.drop(columns=["Timestamp", "Anomaly_Label"]).columns

st.write("Enter feature values:")

input_data = []

# Dynamically create inputs
for col in feature_columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    result, prob = predict(input_data)

    if result == 1:
        st.error("Abnormal detected 🚨")
    else:
        st.success("Normal ✅")

    # 🔥 LangChain explanation
    explanation = explain_prediction(input_data, result)
    st.write("### Explanation:")
    st.write(explanation)