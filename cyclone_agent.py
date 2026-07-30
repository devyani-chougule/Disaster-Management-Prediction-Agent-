from tools.cyclone_tool import cyclone_prediction
from tools.shap_tool import shap_explanation
from tools.rag_tool import get_guidelines



def cyclone_agent(X):


    prediction = cyclone_prediction(X)


    explanation = shap_explanation(
    prediction["model"],
    X
)


    guidelines = get_guidelines(
        "Cyclone safety recommendations"
    )


    return {


        "prediction":prediction,

        "explanation":explanation,

        "guidelines":guidelines

    }