import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graph.disaster_graph import disaster_graph
from agents.router_agent import detect_disaster



# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(
    page_title="AI Disaster Management System",
    page_icon="🌊",
    layout="wide"
)



# ---------------------------
# Header
# ---------------------------

st.title("🌊 AI Disaster Management Agent")


st.markdown(
"""
## 🚨 Intelligent Disaster Prediction & Response Platform

AI based system for:

🌧️ Flood Prediction  
🌀 Cyclone Risk Analysis  
🧠 Explainable AI Decision Making  
📚 Emergency Response Recommendation  

Powered by:
Machine Learning + LangGraph Agents + SHAP + RAG
"""
)


st.divider()



# ---------------------------
# Main Disaster Image
# ---------------------------


st.image(
    "https://images.unsplash.com/photo-1547683905-f686c993aae5",
    caption="AI Based Disaster Monitoring System",
    use_container_width=True
)



# ---------------------------
# Upload CSV
# ---------------------------


uploaded_file = st.file_uploader(
    "📂 Upload Disaster CSV",
    type=["csv"]
)



if uploaded_file:


    df = pd.read_csv(uploaded_file)



    st.subheader(
        "📊 Uploaded Disaster Data"
    )


    st.dataframe(
        df.head(),
        use_container_width=True
    )



    # Dataset information


    c1,c2,c3 = st.columns(3)


    with c1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with c2:

        st.metric(
            "Features",
            df.shape[1]
        )


    with c3:

        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )



    st.divider()



    if st.button(
        "🚨 Analyze Disaster Risk",
        use_container_width=True
    ):



        X=df.copy()



        # Remove target

        if "target" in X.columns:

            X=X.drop(
                "target",
                axis=1
            )



        # Flood date

        if "date" in X.columns:


            X["date"]=pd.to_datetime(
                X["date"]
            )


            X["year"]=X["date"].dt.year
            X["month"]=X["date"].dt.month
            X["day"]=X["date"].dt.day


            X=X.drop(
                "date",
                axis=1
            )



        # Cyclone date


        if "DateTime" in X.columns:


            X["DateTime"]=pd.to_datetime(
                X["DateTime"]
            )


            X["year"]=X["DateTime"].dt.year
            X["month"]=X["DateTime"].dt.month
            X["day"]=X["DateTime"].dt.day


            X=X.drop(
                "DateTime",
                axis=1
            )



        # Encode categorical data


        for col in X.select_dtypes(
            include=["object"]
        ).columns:


            X[col]=(
                X[col]
                .astype("category")
                .cat.codes
            )



        # Disaster Detection


        disaster_type=detect_disaster(X)



        if disaster_type=="Unknown":

            st.error(
                "Unable to detect disaster type"
            )

            st.stop()



        st.success(
            f"🚨 Disaster Detected: {disaster_type}"
        )



        # Dynamic Disaster Image


        if disaster_type=="Flood":

            st.image(
                "https://images.unsplash.com/photo-1547683905-f686c993aae5",
                caption="Flood Risk Monitoring",
                use_container_width=True
            )


        elif disaster_type=="Cyclone":

            st.image(
                "https://images.unsplash.com/photo-1527482797697-8795b05a13fe",
                caption="Cyclone Monitoring",
                use_container_width=True
            )



        with st.spinner(
            "🤖 LangGraph Agents analyzing disaster..."
        ):



            result=disaster_graph.invoke(

                {

                "X":X,

                "disaster_type":disaster_type,

                "agent_result":{},

                "report":""

                }

            )



        report=result["report"]


        agent_data=result["agent_result"]


        prediction=agent_data["prediction"]


        probability=prediction["probability"]

        risk=prediction["risk"]




        # ---------------------------
        # Risk Dashboard
        # ---------------------------


        st.divider()


        st.header(
            "🚨 Disaster Risk Dashboard"
        )



        a,b,c=st.columns(3)



        with a:

            st.metric(
                "Disaster Type",
                disaster_type
            )


        with b:


            if risk=="HIGH":

                display="🔴 HIGH RISK"

            elif risk=="MEDIUM":

                display="🟡 MEDIUM RISK"

            else:

                display="🟢 LOW RISK"



            st.metric(
                "Risk Level",
                display
            )



        with c:

            st.metric(
                "Probability",
                f"{probability:.4f}"
            )





        # Gauge Chart


        fig=go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={
                    "text":"Risk Probability (%)"
                },

                gauge={

                    "axis":{
                        "range":[0,100]
                    }

                }

            )

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )




        # Pie Chart


        st.subheader(
            "📊 Risk Distribution"
        )


        risk_df=pd.DataFrame(

            {

            "Status":[
                "Risk",
                "Safe"
            ],

            "Value":[
                probability,
                1-probability
            ]

            }

        )


        pie=px.pie(

            risk_df,

            values="Value",

            names="Status",

            hole=0.4

        )


        st.plotly_chart(
            pie,
            use_container_width=True
        )




        # ---------------------------
        # SHAP Explanation
        # ---------------------------


        st.header(
            "🧠 AI Explanation (SHAP)"
        )


        explanation=agent_data["explanation"]


        st.text(
            explanation
        )



        shap_rows=[]


        for line in explanation.split("\n"):


            feature=line.split()[0]


            if "increases" in line:

                value=1

            else:

                value=-1



            shap_rows.append(

                {

                "Feature":feature,

                "Impact":value

                }

            )



        shap_df=pd.DataFrame(
            shap_rows
        )


        st.dataframe(
            shap_df,
            use_container_width=True
        )



        # SHAP Chart


        fig=px.bar(

            shap_df,

            x="Feature",

            y="Impact",

            title="Feature Contribution"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )





        # ---------------------------
        # Map
        # ---------------------------


        if "lat" in df.columns and "lon" in df.columns:


            st.header(
                "🗺️ Disaster Location Map"
            )


            st.map(

                df[
                    [
                    "lat",
                    "lon"
                    ]
                ]

            )




        # ---------------------------
        # Report
        # ---------------------------


        st.header(
            "📄 AI Disaster Assessment Report"
        )


        st.markdown(
            report
        )




        # ---------------------------
        # Emergency Panel
        # ---------------------------


        st.header(
            "🚑 Emergency Preparedness Guide"
        )


        x,y,z=st.columns(3)



        with x:

            st.info(
            """
            🎒 Emergency Kit

            ✔ Water
            ✔ Food
            ✔ Medicines
            ✔ Documents
            """
            )



        with y:

            st.warning(
            """
            🏠 Safety Actions

            ✔ Follow alerts
            ✔ Avoid risky zones
            ✔ Move to shelters
            """
            )



        with z:

            st.error(
            """
            🚨 Emergency Response

            ✔ Contact rescue teams
            ✔ Help vulnerable people
            ✔ Follow authorities
            """
            )