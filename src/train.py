from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model