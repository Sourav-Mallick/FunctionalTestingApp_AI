import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="QA Strategy AI Advisor", layout="wide", page_icon="🎯")

# --- DATASET & KNOWLEDGE BASE ---
# Features: Web, Mob, Desk, API, NoCode, Expert, BDD, AI_Heal, NLP, VisualAI, CICD, Parallel, SAP, Cloud, Reporting
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
    "BDD": "Behavior Driven Development. Writing tests in 'Given/When/Then' format so business users can read them.",
    "Self-Healing": "AI that automatically detects UI changes (like a button changing ID) and fixes the test script without human intervention.",
    "NLP": "Natural Language Processing. Ability to write test steps in plain English (e.g., 'Click on the Login button').",
    "Visual AI": "AI that compares screenshots to find layout bugs that traditional code-based tests might miss.",
    "Parallel Execution": "Running multiple tests at the same time to reduce total execution time from hours to minutes."
}

# --- UI DESIGN ---
st.title("🏹 QA Tool Strategy & AI Advisor")
st.markdown("---")

# --- SIDEBAR GLOSSARY ---
st.sidebar.header("📖 Terminology Glossary")
for term, desc in GLOSSARY.items():
    st.sidebar.markdown(f"**{term}**: {desc}")

# --- MAIN INTERFACE: INPUTS ---
st.header("1️⃣ Assessment Parameters")
tab_infra, tab_team, tab_advanced = st.tabs(["🌐 Infrastructure", "👥 Team & Culture", "🤖 Modern Capabilities"])

with tab_infra:
    col_a, col_b = st.columns(2)
    with col_a:
        q_web = st.checkbox("Web Application", value=True, help="Testing browser-based apps")
        q_mob = st.checkbox("Mobile Application", help="Testing iOS/Android native or hybrid apps")
    with col_b:
        q_desk = st.checkbox("Desktop Application", help="Testing Windows or macOS installed apps")
        q_api = st.checkbox("API Testing", help="Validating backend services and microservices")

with tab_team:
    q_skill = st.select_slider("Team Expertise Level:", 
                               options=["Manual / Beginner", "Hybrid", "SDET / Expert"],
                               help="Determines if you need a No-Code tool or a Scripting-heavy framework.")
    q_bdd = st.toggle("Require BDD (Gherkin) Support")
    q_opensource = st.toggle("Prioritize Open Source / $0 License")

with tab_advanced:
    c1, c2 = st.columns(2)
    with c1:
        q_heal = st.checkbox("AI Self-Healing")
        q_nlp = st.checkbox("NLP Authoring (English)")
    with c2:
        q_visual = st.checkbox("Visual AI")
        q_parallel = st.checkbox("Parallel Execution")

# --- WEIGHTING SECTION ---
st.header("2️⃣ Strategic Weighting")
st.info("Adjust the sliders to tell the AI which factors are 'Must-Haves' vs 'Nice-to-Haves'.")
w_col1, w_col2, w_col3 = st.columns(3)
w_plat = w_col1.slider("Platform Priority", 1, 10, 8)
w_skill = w_col2.slider("Team Skill Match", 1, 10, 5)
w_ai = w_col3.slider("AI/Modern Features", 1, 10, 3)

# --- AI ANALYSIS ENGINE ---
if st.button("🚀 EXECUTE AI MATCHMAKING", use_container_width=True):
    # Construct User Vector
    user_vec = [
        1 if q_web else 0, 1 if q_mob else 0, 1 if q_desk else 0, 1 if q_api else 0,
        1 if q_skill == "Manual / Beginner" else 0, 1 if q_skill == "SDET / Expert" else 0,
        1 if q_bdd else 0, 1 if q_heal else 0, 1 if q_nlp else 0, 1 if q_visual else 0,
        1, 1 if q_parallel else 0, 0, 1 if q_mob else 0, 1
    ]

    # Apply Weights
    weights = [w_plat]*4 + [w_skill]*3 + [w_ai]*4 + [5]*4 # Mix of user and fixed weights
    
    # Calculate Weighted Similarity
    results = []
    for tool, t_vec in TOOL_DATA.items():
        w_user = np.array(user_vec) * np.array(weights)
        w_tool = np.array(t_vec) * np.array(weights)
        
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
        st.write("**Why this match?**")
        st.write(f"Based on your preference for **{q_skill}** and your **{w_plat}/10** platform priority, this tool offers the optimal balance of features and ease-of-use.")

    with col_res2:
        fig = px.bar(res_df, x="Match Score", y="Tool", orientation='h', 
                     color="Match Score", color_continuous_scale="Viridis", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison Table")
    st.dataframe(res_df.style.background_gradient(cmap='Blues'), use_container_width=True)
