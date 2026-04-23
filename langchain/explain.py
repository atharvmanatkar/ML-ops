import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def explain_prediction(input_data, prediction):
    api_key = os.getenv("GOOGLE_API_KEY")

    label = "Abnormal" if prediction == 1 else "Normal"

    if not api_key:
        return f"The system predicted '{label}' based on input features."

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Input features: {input_data}
    Prediction: {label}

    Explain in simple and short terms why this prediction occurred.
    """

    response = model.generate_content(prompt)

    return response.text