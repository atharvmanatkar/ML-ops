import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def predict(input_data):
    model = load_model()
    prediction = model.predict([input_data])
    return prediction[0]