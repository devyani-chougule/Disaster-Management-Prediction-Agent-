import joblib


model = joblib.load(
    r"C:\Users\Devyani Chougule\Downloads\flood_prediction_model.pkl"
)


def flood_prediction(X):

    probability = model.predict_proba(X)[0][1]

    if probability < 0.3:
        risk = "LOW"

    elif probability < 0.7:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    return {
        "probability": float(probability),
        "risk": risk,
        "model": model
    }