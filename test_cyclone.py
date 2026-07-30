import joblib
import pandas as pd


# Load cyclone model
model = joblib.load(
    r"C:\Users\Devyani Chougule\Downloads\cyclone_model.pkl"
)


# Test input
sample = pd.DataFrame({
    "Basin": [0],
    "Longitude": [-68.6],
    "Latitude": [34.5],
    "WindSpeed": [35],
    "PressureDrop": [20],
    "Pressure": [990],
    "year": [2003],
    "month": [4],
    "day": [19]
})


# Prediction probability
probability = model.predict_proba(sample)[0][1]


# Risk classification
if probability < 0.3:
    risk = "LOW"

elif probability < 0.7:
    risk = "MEDIUM"

else:
    risk = "HIGH"


print("Cyclone Probability:", round(probability, 2))
print("Cyclone Risk Level:", risk)