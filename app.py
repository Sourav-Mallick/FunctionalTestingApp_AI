import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="QA Strategy AI", layout="wide")
st.title("🏹 Enterprise QA Tool Recommendation Engine")

# --- DATASET ---
data = {
    "Tool": ["Selenium", "Playwright", "Cypress", "Appium", "Katalon", "testRigor", "Tosca"],
    "Web": [1, 1, 1, 0, 1, 1, 1],
    "Mobile": [0, 0, 0, 1, 1, 1, 1],
    "NoCode": [0, 0, 0, 0, 1, 1, 1],
    "AI_Heal": [0, 0, 0, 0, 1, 1, 1],
    "OpenSource": [1, 1, 1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)

# --- UI: 3-COLUMN INPUT ---
st.subheader("📋 Step 1: Define Project Requirements")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Platform**")
    u_web = st.checkbox("Web Apps")
    u_mob = st.checkbox("Mobile Apps")

with col2:
    st.write("**Team & Budget**")
    u_nocode = st.toggle("Require No-Code / Scriptless")
    u_free = st.toggle("Prefer Open Source / Free")

with col3:
    st.write("**Advanced**")
    u_ai = st.checkbox("AI Self-Healing Needed")

# --- ANALYSIS ---
if st.button("🚀 Run Analysis", use_container_width=True):
    user_vec = np.array([[u_web, u_mob, u_nocode, u_ai, u_free]])
    tool_features = df.drop("Tool", axis=1)
    
    # AI Calculation
    scores = cosine_similarity(user_vec, tool_features)[0]
    df['Match %'] = (scores * 100).round(1)
    results = df.sort_values(by="Match %", ascending=False)
    
    st.success(f"### Recommended Tool: {results.iloc[0]['Tool']}")
    
    # Professional Chart
    fig = px.bar(results, x="Match %", y="Tool", color="Match %", orientation='h',
                 title="Compatibility Ranking", color_continuous_scale='Blues')
    st.plotly_chart(fig)