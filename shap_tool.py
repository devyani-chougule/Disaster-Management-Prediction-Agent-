import shap
import numpy as np



def shap_explanation(model, X):


    # Create SHAP explainer

    explainer = shap.TreeExplainer(
        model
    )


    # Calculate SHAP values

    shap_values = explainer.shap_values(
        X
    )


    # Handle different SHAP outputs

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    else:

        values = shap_values[0]



    # Store feature impacts

    explanation = []


    feature_impact = []


    for feature, value in zip(
        X.columns,
        values
    ):

        feature_impact.append(
            (
                feature,
                abs(value),
                value
            )
        )



    # Sort by importance

    feature_impact.sort(
        key=lambda x:x[1],
        reverse=True
    )



    # Take top 10 factors

    for feature, importance, value in feature_impact[:10]:


        if value > 0:

            explanation.append(
                f"{feature} increases disaster risk"
            )

        else:

            explanation.append(
                f"{feature} decreases disaster risk"
            )



    return "\n".join(explanation)