import joblib


cyclone_model = joblib.load(
    r"C:\Users\Devyani Chougule\Downloads\cyclone_model.pkl"
)



def cyclone_prediction(X):

    probability = cyclone_model.predict_proba(X)[0][1]


    if probability < 0.3:
        risk="LOW"

    elif probability <0.7:
        risk="MEDIUM"

    else:
        risk="HIGH"

    return {
    "probability": float(probability),
    "risk": risk
}
