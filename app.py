%%writefile app.py
import streamlit as st
import joblib
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Page Config
st.set_page_config(page_title="Heart Health Predictor & AI Doctor", layout="wide")
st.title("🫀 Heart Health Predictor & AI Medical Assistant")

# Layout Split
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN: DECISION TREE MODEL PREDICTION ---
with col1:
    st.header("1. Heart Disease Prediction")
    st.write("Enter your medical details:")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    if st.button("Predict Disease Risk"):
        try:
            # REAL MODEL LOAD KAR RAHE HAIN (.pkl file se)
            model = joblib.load('heart_model.pkl')
            features = np.array([[age, bp, chol]])
            prediction = model.predict(features)
            
            if prediction[0] == 1:
                st.error("⚠️ High Risk of Heart Disease detected.")
            else:
                st.success("✅ Low Risk of Heart Disease detected.")
        except Exception as e:
            st.error("Model file 'heart_model.pkl' not found! Please check GitHub repository.")

# --- RIGHT COLUMN: AI DOCTOR CHATBOT ---
with col2:
    st.header("2. AI Medical Assistant")
    st.write("Ask the AI Doctor to understand medical terms:")

    user_query = st.text_input("Your Question:")

    if st.button("Ask Doctor AI"):
        if not user_query:
            st.warning("Please type a question!")
        else:
            with st.spinner("Doctor thinking..."):
                try:
                    # Streamlit Secrets se API Key
                    llm = ChatGroq(
                        model='llama-3.1-8b-instant',
                        temperature=0.7,
                        api_key=st.secrets["GROQ_API_KEY"]
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful medical doctor. Answer health questions in very simple language with clear examples."),
                        ("human", "{user_question}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"user_question": user_query})
                    st.info(response)
                except Exception as e:
                    st.error(f"Error: {e}")
