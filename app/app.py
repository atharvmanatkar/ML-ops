import streamlit as st
from src.predict import predict

st.title("Anomaly Detection System")

st.write("Enter values:")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    result = predict([feature1, feature2, feature3])

    if result == 1:
        st.error("Abnormal detected 🚨")
    else:
        st.success("Normal ✅")