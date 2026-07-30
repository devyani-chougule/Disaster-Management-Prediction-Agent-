import joblib
import pandas as pd
import shap
import requests

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
model = joblib.load(
    r"C:\Users\Devyani Chougule\Downloads\cyclone_model.pkl"
)
sample = pd.DataFrame({

    "Basin":[0],
    "Longitude":[-68.6],
    "Latitude":[34.5],
    "WindSpeed":[35],
    "PressureDrop":[20],
    "Pressure":[990],
    "year":[2003],
    "month":[4],
    "day":[19]

})
probability = model.predict_proba(sample)[0][1]


if probability < 0.3:
    risk="LOW"

elif probability < 0.7:
    risk="MEDIUM"

else:
    risk="HIGH"


print(probability)
print(risk)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)


if isinstance(shap_values,list):
    values = shap_values[1][0]
else:
    values = shap_values[0]


reasons = []


for feature,value in zip(
    sample.columns,
    values
):

    if value > 0:
        reasons.append(
            f"{feature} increases cyclone risk"
        )

    else:
        reasons.append(
            f"{feature} reduces cyclone risk"
        )


reason_text = "\n".join(reasons)


print(reason_text)
loader = TextLoader(
    r"C:\Users\Devyani Chougule\Desktop\NLP (Disaster Managment)\cycloneinfo.txt"
)


docs = loader.load()


embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = Chroma.from_documents(
    docs,
    embeddings
)
results = db.similarity_search(
    "cyclone safety recommendations",
    k=2
)


rag_context = "\n".join(
    [doc.page_content for doc in results]
)
prompt = f"""

You are an AI disaster management assistant.

Create a cyclone risk assessment report.

Prediction:

Cyclone Probability:
{probability:.2f}

Risk Level:
{risk}


Machine Learning Explanation:

{reason_text}


Retrieved Cyclone Guidelines:

{rag_context}


Create sections:

1. Risk Level
2. Possible Reasons
3. Safety Recommendations
4. Emergency Actions

Write a professional disaster report.

"""
response = requests.post(
"http://localhost:11434/api/generate",

json={
"model":"phi3",
"prompt":prompt,
"stream":False
}

)


report = response.json()["response"]


print(report)