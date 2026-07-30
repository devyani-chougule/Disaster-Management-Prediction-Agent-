from tools.flood_tool import flood_prediction
from tools.shap_tool import shap_explanation
from tools.rag_tool import get_guidelines



def flood_agent(X):


    # Flood ML Prediction

    prediction = flood_prediction(
        X
    )



    # SHAP Explanation

    explanation = shap_explanation(
        prediction["model"],
        X
    )



    # Retrieve Disaster Guidelines

    guidelines = get_guidelines(
        "Flood disaster safety recommendations and emergency actions"
    )



    return {

        "prediction": prediction,

        "explanation": explanation,

        "guidelines": guidelines

    }