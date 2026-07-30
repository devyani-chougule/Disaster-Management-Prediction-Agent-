from tools.flood_tool import flood_prediction
from tools.cyclone_tool import cyclone_prediction
from tools.ollama_tool import generate_report
from tools.shap_tool import shap_explanation
from tools.rag_tool import get_guidelines



def disaster_agent(disaster_type, X):


    # -----------------------------
    # Select Prediction Tool
    # -----------------------------

    if disaster_type == "Flood":

        result = flood_prediction(X)


    elif disaster_type == "Cyclone":

        result = cyclone_prediction(X)


    else:

        return "Invalid disaster type"



    model = result["model"]



    # -----------------------------
    # SHAP Explanation
    # -----------------------------

    explanation = shap_explanation(
        model,
        X
    )



    # -----------------------------
    # RAG Retrieval
    # -----------------------------

    guidelines = get_guidelines(
        f"""
        {disaster_type} disaster guidelines.

        Risk Level:
        {result['risk']}

        Provide safety recommendations,
        emergency actions,
        and response procedures.
        """
    )



    # -----------------------------
    # Ollama Prompt
    # -----------------------------

    prompt = f"""

You are a Disaster Management Assistant.

Create a professional disaster assessment report.


Rules:

- Do not add author names.
- Do not add signatures.
- Do not mention AI.
- Do not create fictional experts.
- Do not add placeholders.
- Use only provided SHAP factors.
- Do not invent environmental explanations.



Disaster Type:

{disaster_type}



Prediction Result:


Probability:

{result['probability']:.4f}



Risk Level:

{result['risk']}



SHAP Important Factors:


{explanation}



Retrieved Disaster Guidelines:


{guidelines}



Generate report using this format:


## Disaster Risk Assessment Report


### 1. Risk Level Analysis

Explain the predicted risk level using probability.


### 2. Possible Reasons

Explain important environmental factors from SHAP.


### 3. Safety Recommendations

Give practical safety steps for citizens and authorities.


### 4. Emergency Actions

Give immediate response actions.


Keep the report professional and concise.

"""



    # -----------------------------
    # Generate Report
    # -----------------------------

    report = generate_report(
        prompt
    )


    return report