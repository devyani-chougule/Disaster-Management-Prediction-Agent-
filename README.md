#  AI Disaster Management Agent

An AI-powered disaster risk assessment system that predicts flood and cyclone risks using Machine Learning, Explainable AI, Agentic AI, and Retrieval-Augmented Generation (RAG).

The system analyzes environmental disaster data, predicts risk levels, explains prediction factors, retrieves safety guidelines, and generates automated disaster assessment reports.

---

#  Project Overview

Natural disasters require fast prediction and effective response planning. This project builds an intelligent disaster management assistant that combines:

- Machine Learning based risk prediction
- Multi-Agent AI workflow using LangGraph
- Explainable AI using SHAP
- Retrieval-Augmented Generation (RAG)
- LLM-based report generation
- Interactive Streamlit dashboard

The system currently supports:

 Flood Risk Prediction  
 Cyclone Risk Prediction  
 Automated Disaster Reports  
 Safety Recommendations  
 Risk Explanation

---

#  System Architecture


User Uploads Disaster Data (CSV)

↓

Data Preprocessing

↓

Disaster Detection Agent

↓

LangGraph Multi-Agent Workflow

↓

 ├── Flood Prediction Agent

 ├── Cyclone Prediction Agent

 ├── SHAP Explanation Agent

 ├── RAG Safety Guideline Retrieval

 └── Report Generation Agent

↓

AI Disaster Assessment Report


---

#  Features

## 1. Disaster Risk Prediction

Machine learning models analyze environmental parameters and predict disaster probability.

Supported models:

- Flood → LightGBM Classifier
- Cyclone → Machine Learning Classification Model


---

## 2. Agentic AI Workflow

Implemented using LangGraph.

Agents:

### Router Agent
Detects the disaster type and routes the request.

### Flood Agent
Performs flood risk prediction and explanation.

### Cyclone Agent
Performs cyclone risk prediction and explanation.

### Report Agent
Generates final disaster assessment reports.


---

## 3. Explainable AI (SHAP)

SHAP (SHapley Additive Explanations) explains why the model predicted a particular risk.

Example factors:

- Rainfall
- NDWI
- Elevation
- Longitude
- Latitude
- Environmental conditions


---

## 4. RAG Based Disaster Guidelines

Retrieves relevant disaster safety information from the knowledge base.

Provides:

- Emergency actions
- Safety recommendations
- Response procedures


---

## 5. AI Report Generation

Uses Ollama LLM with prompt engineering to generate structured reports containing:

- Risk Level Analysis
- Possible Causes
- Safety Recommendations
- Emergency Actions


---

# 🛠️ Technology Stack

## Programming Language

- Python


## Machine Learning

- LightGBM
- Scikit-learn
- Joblib


## Explainable AI

- SHAP


## Generative AI

- Ollama LLM
- Prompt Engineering
- RAG


## Agent Framework

- LangGraph


## Data Processing

- Pandas
- NumPy


## Visualization & UI

- Streamlit
- Plotly


---

# 📂 Project Files

Since files are uploaded separately, the repository contains:


```
AI-Disaster-Management-Agent

│
├── app1.py
│       Streamlit application interface
│
├── disaster_graph.py
│       LangGraph workflow management
│
├── flood_agent.py
│       Flood prediction agent
│
├── cyclone_agent.py
│       Cyclone prediction agent
│
├── router_agent.py
│       Disaster type detection
│
├── report_agent.py
│       Final report generation
│
├── flood_tool.py
│       Flood ML prediction pipeline
│
├── cyclone_tool.py
│       Cyclone ML prediction pipeline
│
├── shap_tool.py
│       SHAP explanation module
│
├── rag_tool.py
│       Disaster guideline retrieval
│
├── ollama_tool.py
│       LLM report generation
│
├── Flood_Risk_Model_Training.ipynb
│       Model training notebook
│
├── requirements.txt
│       Required Python libraries
│
└── README.md

```

---

#  Installation


Clone repository:

```bash
git clone https://github.com/yourusername/AI-Disaster-Management-Agent.git
```


Install dependencies:

```bash
pip install -r requirements.txt
```


Run application:

```bash
streamlit run app1.py
```


---

#  Model Performance

Flood Risk Prediction Model:

Algorithm:

LightGBM Classifier


Evaluation:

ROC-AUC Score:

```
0.99
```


---

#  Application Screenshots

<img width="1295" height="533" alt="Screenshot 2026-07-30 131625" src="https://github.com/user-attachments/assets/410d2f23-017d-4f6f-8bb5-57b81e599d3b" />
<img width="1665" height="792" alt="Screenshot 2026-07-30 131656" src="https://github.com/user-attachments/assets/60690a0c-a8c8-4031-b2bc-5d402de89ffd" />
<img width="1845" height="682" alt="Screenshot 2026-07-30 131749" src="https://github.com/user-attachments/assets/3e50da4c-6a3c-42dd-9042-949ed403294f" />
<img width="1746" height="818" alt="Screenshot 2026-07-30 131736" src="https://github.com/user-attachments/assets/0c0d7831-5787-4462-ad29-107bdeffdc07" />
<img width="1828" height="721" alt="Screenshot 2026-07-30 131816" src="https://github.com/user-attachments/assets/d00c3520-b028-4203-a604-418384f7e98f" />
<img width="1800" height="448" alt="Screenshot 2026-07-30 131825" src="https://github.com/user-attachments/assets/75d062ef-cea8-47b1-8601-5cf5d314b9ad" />


Example:

- Risk Dashboard
- SHAP Explanation
- Disaster Report
- Location Map


---

#  Future Improvements

- Real-time weather API integration
- Satellite image based disaster detection
- Deep Learning based flood segmentation
- Live disaster alert system
- Mobile application integration


---

#  Author

Devyani Chougule

AI/ML Engineering Student
