from imblearn.over_sampling import RandomOverSampler

def preprocess_data(df):
    # Encode labels
    df["Anomaly_Label"] = df["Anomaly_Label"].map({
        "Normal": 0,
        "Abnormal": 1
    })

    # Drop unnecessary columns
    X = df.drop(columns=["Timestamp", "Anomaly_Label"])
    y = df["Anomaly_Label"]

    # Handle imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled