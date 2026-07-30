import joblib
import pandas as pd
import shap


# Load model
model = joblib.load(
    r"C:\Users\Devyani Chougule\Downloads\cyclone_model.pkl"
)


# Same input used for prediction

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


# Create SHAP explainer

explainer = shap.TreeExplainer(model)


# Calculate SHAP values

shap_values = explainer.shap_values(sample)


print("SHAP Explanation")
print("----------------")


# Handle new SHAP output format

if isinstance(shap_values, list):
    values = shap_values[1][0]
else:
    values = shap_values[0]


print("SHAP Explanation")
print("----------------")


for feature, value in zip(
    sample.columns,
    values
):
    print(
        feature,
        ":",
        round(value,4)
    )