import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="QA Tool Strategy Advisor", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 Enterprise QA Tool Recommendation Engine")
st.write("Provide your project constraints below for a weighted mathematical analysis of the best-fit automation frameworks.")

# --- THE DATASET (Expanded with 12 features per tool) ---
# Order: Web, Mobile, Desktop, API, NoCode, Expert, BDD, AI_SelfHeal, NLP, VisualAI, CICD, Parallel
tools_dict = {
    "Selenium":      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    "Playwright":    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    "Cypress":       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    "Appium":        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
    "Katalon":       [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    "testRigor":     [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    "Tricentis Tosca": [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
}

# --- UI INPUTS ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Infrastructure & Apps")
        q_platforms = st.multiselect("Target Platforms:", ["Web", "Mobile", "Desktop", "API"], help="Select all that apply to your project.")
        q_budget = st.radio("Licensing Model:", ["Open Source", "Enterprise/Commercial"], horizontal=True)
        
    with col2:
        st.subheader("👥 Team & Skillset")
        q_skill = st.select_slider("Required Authoring Style:", options=["Scripting (Code)", "Hybrid", "No-Code (Plain English)"])
        q_bdd = st.checkbox("BDD Support Required (Gherkin/Cucumber)")

    st.divider()
    
    with st.expander("🛠️ Advanced Feature Requirements"):
        c1, c2, c3 = st.columns(3)
        with c1:
            q_ai = st.checkbox("AI Self-Healing")
            q_nlp = st.checkbox("NLP / English Test Creation")
        with c2:
            q_visual = st.checkbox("Visual Regression AI")
            q_parallel = st.checkbox("Parallel Execution")
        with c3:
            q_cicd = st.checkbox("Native CI/CD Integration")

# --- ANALYTICS LOGIC ---
if st.button("🚀 Execute Strategy Analysis", use_container_width=True):
    # Convert UI to Vector
    u_vec = [
        1 if "Web" in q_platforms else 0,
        1 if "Mobile" in q_platforms else 0,
        1 if "Desktop" in q_platforms else 0,
        1 if "API" in q_platforms else 0,
        1 if q_skill == "No-Code (Plain English)" else 0,
        1 if q_skill == "Scripting (Code)" else 0,
        1 if q_bdd else 0,
        1 if q_ai else 0,
        1 if q_nlp else 0,
        1 if q_visual else 0,
        1 if q_cicd else 0,
        1 if q_parallel else 0
    ]

    # Weighted Scoring Calculation
    results = []
    for tool, t_vec in tools_dict.items():
        score = cosine_similarity([u_vec], [t_vec])[0][0]
        results.append({"Tool": tool, "Match Score": round(score * 100, 1)})
    
    res_df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)

    # --- RESULTS UI ---
    st.divider()
    top_tool = res_df.iloc[0]['Tool']
    
    c_res1, c_res2 = st.columns([1, 2])
    
    with c_res1:
        st.metric(label="Primary Recommendation", value=top_tool)
        st.write(f"The model found **{top_tool}** to be the highest mathematical match for your project constraints.")
        
    with c_res2:
        fig = px.bar(res_df, x="Match Score", y="Tool", orientation='h', 
                     color="Match Score", color_continuous_scale="GnBu",
                     text_auto=True)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Detailed Capability Comparison")
    st.table(res_df)
