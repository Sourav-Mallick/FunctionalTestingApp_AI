import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Weighted QA AI Advisor", layout="wide")

# --- DATASET ---
# Features: Web, Mob, Desk, API, NoCode, Expert, BDD, AI_Heal, NLP, VisualAI, CICD, Parallel, SAP, Cloud
tools_dict = {
    "Selenium":      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    "Playwright":    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    "Cypress":       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "Appium":        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    "Katalon":       [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    "testRigor":     [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    "Tricentis Tosca": [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
}

st.title("🏹 Weighted AI Tool Recommendation Engine")
st.write("This model uses weighted vector similarity to match tools based on priority levels.")

# --- STEP 1: DEFINE REQUIREMENTS ---
st.header("1️⃣ Define Requirements")
tabs = st.tabs(["Platforms", "Team", "Advanced"])

with tabs[0]:
    c1, c2 = st.columns(2)
    q_web = c1.checkbox("Web", value=True)
    q_mob = c1.checkbox("Mobile")
    q_desk = c2.checkbox("Desktop")
    q_api = c2.checkbox("API")

with tabs[1]:
    q_skill = st.select_slider("Skill Level:", ["No-Code", "Hybrid", "Scripting"])
    q_bdd = st.toggle("BDD Required")

with tabs[2]:
    q_ai = st.checkbox("AI Self-Healing")
    q_nlp = st.checkbox("NLP Authoring")
    q_cicd = st.checkbox("CI/CD Integration")

# --- STEP 2: DEFINE WEIGHTS (The "AI Tuning") ---
st.header("2️⃣ Set Feature Priority")
st.info("How much weight should the AI give to each category?")
col_w1, col_w2, col_w3 = st.columns(3)

w_platform = col_w1.slider("Platform Importance", 1.0, 5.0, 3.0)
w_skills = col_w2.slider("Team Skill Match", 1.0, 5.0, 2.0)
w_modern = col_w3.slider("AI/Modern Features", 1.0, 5.0, 1.0)

# --- ANALYSIS ---
if st.button("🚀 RUN WEIGHTED ANALYSIS"):
    # Create User Vector
    u_vec = np.array([
        1 if q_web else 0, 1 if q_mob else 0, 1 if q_desk else 0, 1 if q_api else 0,
        1 if q_skill == "No-Code" else 0, 1 if q_skill == "Scripting" else 0,
        1 if q_bdd else 0, 1 if q_ai else 0, 1 if q_nlp else 0, 0, # VisualPlaceholder
        1 if q_cicd else 0, 0, 0, 0 # Padding for remaining features
    ])

    # Create Weight Vector
    # We apply weights to specific indices of the vector
    weights = np.array([
        w_platform, w_platform, w_platform, w_platform, # Platforms
        w_skills, w_skills, w_skills, # Skills
        w_modern, w_modern, w_modern, w_modern, w_modern, w_modern, w_modern # Modern/DevOps
    ])

    # Calculate Weighted Scores
    results = []
    for tool, t_vec in tools_dict.items():
        # Weighted math: Multiply vectors by weights before similarity
        w_u_vec = u_vec * weights
        w_t_vec = np.array(t_vec) * weights
        
        score = cosine_similarity([w_u_vec], [w_t_vec])[0][0]
        results.append({"Tool": tool, "Match Score": round(score * 100, 1)})

    res_df = pd.DataFrame(results).sort_values("Match Score", ascending=False)
    
    # UI Results
    st.divider()
    st.success(f"### Top Recommendation: {res_df.iloc[0]['Tool']}")
    
    fig = px.bar(res_df, x="Match Score", y="Tool", orientation='h', color="Match Score")
    st.plotly_chart(fig, use_container_width=True)
