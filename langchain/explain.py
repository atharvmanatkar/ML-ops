from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

def explain_prediction(input_data, prediction):
    llm = ChatOpenAI(
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
    Input data: {input_data}
    Prediction: {prediction}

    Explain in simple terms why this prediction occurred.
    """

    response = llm([HumanMessage(content=prompt)])

    return response.content