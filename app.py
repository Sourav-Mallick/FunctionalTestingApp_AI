import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic QA Advisor", layout="wide", page_icon="🎯")

# --- DATASET & DESCRIPTIONS ---
# We now include 15 technical features for each tool
tools_dict = {
    "Selenium":      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    "Playwright":    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    "Cypress":       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "Appium":        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    "Katalon":       [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    "testRigor":     [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    "Tricentis Tosca": [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
}

descriptions = {
    "Selenium": "The open-source pioneer. Best for high-code, customized web automation.",
    "Playwright": "Modern, fast, and extremely reliable. The new favorite for JS-heavy web apps.",
    "Cypress": "Dev-friendly and great for front-end integration. Limited by single-tab execution.",
    "Appium": "The gold standard for mobile automation. Requires high technical expertise.",
    "Katalon": "Comprehensive all-in-one low-code platform. Great for teams moving from manual testing.",
    "testRigor": "AI-First. Allows manual testers to write complex tests in plain English.",
    "Tricentis Tosca": "The enterprise choice for complex end-to-end business process testing (SAP, Web, Mobile)."
}

# --- UI DESIGN ---
st.title("🎯 Strategic Functional Testing Tool Advisor")
st.markdown("---")

# Categories using Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Infrastructure", "👥 Team & Skills", "🤖 Advanced AI", "⚙️ DevOps"])

with tab1:
    st.subheader("Application Tech Stack")
    col_a, col_b = st.columns(2)
    with col_a:
        q_web = st.checkbox("Web Application", value=True)
        q_mob = st.checkbox("Mobile (Native/Hybrid)")
        q_desk = st.checkbox("Desktop (Windows/Mac)")
    with col_b:
        q_api = st.checkbox("API / Microservices")
        q_sap = st.checkbox("Enterprise ERP (SAP/Salesforce)")

with tab2:
    st.subheader("Personnel & Expertise")
    q_skill = st.select_slider("Select Team Technical Level:", 
                               options=["Manual Only", "Hybrid (Some Coding)", "Full SDET (Coders)"])
    q_lang = st.selectbox("Primary Language Preference:", ["JavaScript", "Python", "Java", "C#", "No Preference"])
    q_bdd = st.toggle("Require BDD (Gherkin/Cucumber) Support")

with tab3:
    st.subheader("Modern Automation Capabilities")
    col_c, col_d = st.columns(2)
    with col_c:
        q_heal = st.checkbox("AI Self-Healing (Auto-update locators)", help="Reduces maintenance by 70%")
        q_nlp = st.checkbox("Plain English Test Authoring")
    with col_d:
        q_visual = st.checkbox("Visual Regression AI", help="Pixel-perfect screenshot comparisons")
        q_healing = st.checkbox("Auto-Test Data Generation")

with tab4:
    st.subheader("Operational Execution")
    q_cicd = st.checkbox("Native CI/CD Pipeline Integration")
    q_parallel = st.checkbox("Parallel Execution (Running 50+ tests simultaneously)")
    q_cloud = st.checkbox("Managed Cloud Device Farm Support")

# --- ANALYSIS ENGINE ---
if st.button("🚀 GENERATE RECOMMENDATION REPORT", use_container_width=True):
    # Mapping UI to 15-point Vector
    u_vec = [
        1 if q_web else 0, 1 if q_mob else 0, 1 if q_desk else 0, 1 if q_api else 0,
        1 if q_skill == "Manual Only" else 0, 1 if q_skill == "Full SDET (Coders)" else 0,
        1 if q_bdd else 0, 1 if q_heal else 0, 1 if q_nlp else 0, 1 if q_visual else 0,
        1 if q_cicd else 0, 1 if q_parallel else 0, 1 if q_sap else 0,
        1 if q_skill != "Full SDET (Coders)" else 0, # Low-code preference
        1 if q_cloud else 0
    ]

    # Weighted Cosine Similarity
    results = []
    for tool, t_vec in tools_dict.items():
        score = cosine_similarity([u_vec], [t_vec])[0][0]
        results.append({"Tool": tool, "Match Score": round(score * 100, 1)})
    
    res_df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
    
    # Visualizing Results
    st.divider()
    top_tool = res_df.iloc[0]['Tool']
    
    col_res1, col_res2 = st.columns([1, 1.5])
    
    with col_res1:
        st.header("🏆 Recommended Path")
        st.success(f"### {top_tool}")
        st.write(descriptions[top_tool])
        st.metric("Compatibility Score", f"{res_df.iloc[0]['Match Score']}%")
        
    with col_res2:
        fig = px.bar(res_df.head(5), x="Match Score", y="Tool", orientation='h', 
                     color="Match Score", color_continuous_scale="RdYlGn",
                     title="Top 5 Framework Match Percentage")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Comparative Benchmarking")
    st.dataframe(res_df.style.highlight_max(axis=0), use_container_width=True)
