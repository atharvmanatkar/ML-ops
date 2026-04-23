# Anomaly Detection MLOps Project

## Overview

This project implements an end-to-end MLOps pipeline for anomaly detection using machine learning.

## Features

- Data preprocessing with imbalance handling
- Model training using Random Forest
- DVC for data & model versioning
- CI/CD using GitHub Actions
- Streamlit UI for predictions
- Optional LLM-based explanation (LangChain)

## Tech Stack

- Python, Scikit-learn
- DVC
- GitHub Actions
- Streamlit
- LangChain (optional)

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run pipeline

python main.py

### 3. Run app

streamlit run app/app.py
