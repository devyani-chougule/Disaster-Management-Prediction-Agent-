from tools.ollama_tool import generate_report



def report_agent(
    disaster_type,
    result
):


    prompt=f"""

You are a disaster management expert.

Create a professional disaster report.

Disaster Type:

{disaster_type}


Prediction:

{result["prediction"]}


SHAP Factors:

{result["explanation"]}


Guidelines:

{result["guidelines"]}



Format:

1. Risk Analysis

2. Possible Reasons

3. Safety Recommendations

4. Emergency Actions


Do not invent information.

"""


    return generate_report(prompt)