import pickle
import numpy as np

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def predict(input_data):
    model = load_model()
    input_array = np.array(input_data).reshape(1, -1)

    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0].max()

    return pred, prob