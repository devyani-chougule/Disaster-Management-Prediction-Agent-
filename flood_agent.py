from tools.flood_tool import flood_prediction
from tools.shap_tool import shap_explanation
from tools.rag_tool import get_guidelines



def flood_agent(X):



    prediction = flood_prediction(
        X
    )




    explanation = shap_explanation(
        prediction["model"],
        X
    )




    guidelines = get_guidelines(
        "Flood disaster safety recommendations and emergency actions"
    )



    return {

        "prediction": prediction,

        "explanation": explanation,

        "guidelines": guidelines

    }
