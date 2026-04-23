from src.data_ingestion import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def run_pipeline():
    print("Loading data...")
    df = load_data("data/raw/Ai-data.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Final Metrics:", metrics)

if __name__ == "__main__":
    run_pipeline()