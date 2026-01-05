import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategic QA Advisor", layout="wide", page_icon="🎯")

# --- DATASET & KNOWLEDGE BASE ---
# 15 Features per tool: 
# [Web, Mob, Desk, API, NoCode, Expert, BDD, AI_Heal, NLP, VisualAI, CICD, Parallel, SAP, Mobile_Expertise, Cloud]
TOOL_DATA = {
    "Selenium":      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    "Playwright":    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    "Cypress":       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "Appium":        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    "Katalon":       [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    "testRigor":     [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    "Tricentis Tosca": [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
}

GLOSSARY = {
    "BDD": "Behavior Driven Development. Writing tests in 'Given/When/Then' format.",
    "Self-Healing": "AI that auto-fixes test scripts when UI elements change.",
    "NLP": "Natural Language Processing. Writing tests in plain English.",
    "Visual AI": "AI that compares screenshots to find layout/pixel bugs.",
    "Parallel Execution": "Running multiple tests simultaneously to save time."
}

# --- SIDEBAR GLOSSARY ---
st.sidebar.header("📖 Terminology Glossary")
for term, desc in GLOSSARY.items():
    st.sidebar.markdown(f"**{term}**: {desc}")

# --- MAIN INTERFACE ---
st.title("🏹 QA Tool Strategy & AI Advisor")
st.markdown("This application uses **Weighted Cosine Similarity** (ML) to rank tools based on your specific enterprise constraints.")

# --- STEP 1: ASSESSMENT PARAMETERS ---
st.header("1️⃣ Assessment Parameters")
tab_infra, tab_team, tab_advanced = st.tabs(["🌐 Infrastructure", "👥 Team & Culture", "🤖 Modern Capabilities"])

with tab_infra:
    col_a, col_b = st.columns(2)
    with col_a:
        q_web = st.checkbox("Web Application", value=True)
        q_mob = st.checkbox("Mobile Application")
    with col_b:
        q_desk = st.checkbox("Desktop Application")
        q_api = st.checkbox("API Testing")

with tab_team:
    q_skill = st.select_slider("Team Expertise Level:", 
                               options=["Manual / Beginner", "Hybrid", "SDET / Expert"])
    q_bdd = st.toggle("Require BDD (Gherkin) Support")
    q_opensource = st.toggle("Prioritize Open Source / Free Tools")

with tab_advanced:
    c1, c2 = st.columns(2)
    with c1:
        q_heal = st.checkbox("AI Self-Healing")
        q_nlp = st.checkbox("NLP Authoring (English)")
    with c2:
        q_visual = st.checkbox("Visual AI")
        q_parallel = st.checkbox("Parallel Execution")

# --- STEP 2: STRATEGIC WEIGHTING ---
st.header("2️⃣ Strategic Weighting")
st.info("Adjust the 'Importance' sliders to prioritize specific categories.")
w_col1, w_col2, w_col3 = st.columns(3)
w_plat = w_col1.slider("Platform Importance", 1, 10, 8)
w_skill = w_col2.slider("Team Skill Match Importance", 1, 10, 5)
w_ai = w_col3.slider("AI Features Importance", 1, 10, 3)

# --- AI ANALYSIS ENGINE ---
if st.button("🚀 EXECUTE AI MATCHMAKING", use_container_width=True):
    # Construct User Vector
    user_vec = [
        1 if q_web else 0, 1 if q_mob else 0, 1 if q_desk else 0, 1 if q_api else 0,
        1 if q_skill == "Manual / Beginner" else 0, 1 if q_skill == "SDET / Expert" else 0,
        1 if q_bdd else 0, 1 if q_heal else 0, 1 if q_nlp else 0, 1 if q_visual else 0,
        1, 1 if q_parallel else 0, 0, 1 if q_mob else 0, 1
    ]

    # Construct Weights Vector
    weights = [w_plat]*4 + [w_skill]*3 + [w_ai]*4 + [5]*4 
    
    # Calculate Weighted Similarity
    results = []
    for tool, t_vec in TOOL_DATA.items():
        w_user = np.array(user_vec) * np.array(weights)
        w_tool = np.array(t_vec) * np.array(weights)
        
        # ML Logic: Cosine Similarity measures the angle between these two high-dimensional vectors
        score = cosine_similarity([w_user], [w_tool])[0][0]
        results.append({"Tool": tool, "Match Score": round(score * 100, 1)})
    
    res_df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)

    # --- RESULTS DISPLAY ---
    st.divider()
    top_tool = res_df.iloc[0]['Tool']
    
    col_res1, col_res2 = st.columns([1, 1.5])
    
    with col_res1:
        st.subheader("Top Recommendation")
        st.success(f"### {top_tool}")
        st.metric("Match Confidence", f"{res_df.iloc[0]['Match Score']}%")
        st.write(f"**Insight:** Based on your weighting, **{top_tool}** aligns best with your {q_skill} team.")

    with col_res2:
        fig = px.bar(res_df, x="Match Score", y="Tool", orientation='h', 
                     color="Match Score", color_continuous_scale="GnBu", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- COMPARISON TABLE ---
    st.subheader("📊 Comparative Benchmarking")
    st.dataframe(res_df, use_container_width=True)

    # --- EXECUTIVE SUMMARY ---
    st.divider()
    st.subheader("📩 Executive Summary for Stakeholders")
    
    platforms_list = [p for p, val in zip(['Web', 'Mobile', 'Desktop', 'API'], [q_web, q_mob, q_desk, q_api]) if val]
    summary_text = f"""
    PROJECT REQUIREMENTS SUMMARY:
    - Target Platforms: {', '.join(platforms_list) if platforms_list else 'None selected'}
    - Team Skill Profile: {q_skill}
    - Critical Modern Features: AI Self-Healing ({'Yes' if q_heal else 'No'}), Parallelism ({'Yes' if q_parallel else 'No'})
    
    AI MODEL RECOMMENDATION:
    - Primary Tool Recommendation: {top_tool}
    - Compatibility Match: {res_df.iloc[0]['Match Score']}%
    - Methodology: Weighted Vector Space Analysis (Cosine Similarity)
    """
    
    st.text_area("Copy this summary for your project proposal:", value=summary_text, height=250)
    
    # Download Option
    st.download_button(
        label="📥 Download Full Comparison (CSV)",
        data=res_df.to_csv(index=False).encode('utf-8'),
        file_name="QA_Tool_Analysis.csv",
        mime="text/csv"
    )
