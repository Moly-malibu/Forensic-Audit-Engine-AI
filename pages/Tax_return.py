# app_tax_fraud.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="Tax Fraud Detector AI", page_icon="chart_with_upwards_trend", layout="centered")

# Título con estilo
st.markdown("""
<style>
    .main {background-color: #f8f9fa; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    .title {font-size: 2.8rem; color: #1E3A8A; font-weight: bold; text-align: center; margin-bottom: 1rem;}
    .subtitle {font-size: 1.2rem; color: #475569; text-align: center; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">Tax Fraud Detector AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect tax return fraud with ML</p>', unsafe_allow_html=True)

# === MODELO ENTRENADO (simulado con datos reales) ===
@st.cache_resource
def load_model():
    data = {
        'income': [45000, 120000, 80000, 30000, 200000, 55000, 90000, 150000, 60000, 180000],
        'deductions': [12000, 45000, 18000, 8000, 90000, 11000, 30000, 70000, 15000, 85000],
        'tax_credits': [2000, 8000, 3000, 1000, 15000, 2500, 4000, 12000, 2200, 14000],
        'is_fraud': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# === INPUTS DEL USUARIO ===
st.markdown("### Enter declaration details")
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income($)", min_value=0, value=75000, step=1000)
    deductions = st.number_input("Total Deductions ($)", min_value=0, value=15000, step=500)
with col2:
    credits = st.number_input("Tax Credits($)", min_value=0, value=3000, step=100)

# === PREDICCIÓN ===
if st.button("Detect Fraud", type="primary"):
    prob = model.predict_proba([[income, deductions, credits]])[0][1]
    risk = "HIGHT" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"
    
    st.markdown(f"""
    <div style="padding: 1.5rem; background-color: {color}; color: white; border-radius: 12px; text-align: center; font-size: 1.5rem; font-weight: bold;">
        FRAUD RISK: <span style="font-size: 2rem;">{risk}</span><br>
        Probabilidad: <span style="font-size: 1.8rem;">{prob*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    if risk == "HIGHT":
        st.error("Recomended: In-Depth Audit | Posible 1099-NEC falso")
    elif risk == "MEDIUM":
        st.warning("Review deduccions | Possible inflated HSA")
    else:
        st.success("Cleam Declaration  | low risk")
        
        



# --------------------------------------------------------------
#  Tax Fraud Detector AI – Full Interactive Dashboard
#  
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    # page_title="Tax Fraud Detector AI",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Custom CSS ------------------
st.markdown(
    """
<style>
    .main {background:#fafafa; padding:2rem; border-radius:18px; box-shadow:0 6px 20px rgba(0,0,0,0.1);}
    .title {font-size:3rem; font-weight:800; color:#1d4ed8; text-align:center;}
    .subtitle {font-size:1.3rem; color:#475569; text-align:center; margin-bottom:2rem;}
    .metric-box {background:#e0e7ff; padding:1rem; border-radius:12px; text-align:center; font-weight:600;}
    .footer {margin-top:4rem; font-size:0.95rem; color:#6b7280; text-align:center;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------ Header ------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">AI is Working for your saved</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Real‑time fraud risk scoring with Random Forest + Interactive Analytics</p>',
    unsafe_allow_html=True,
)

# ------------------ Train Model (Cached) ------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    income = np.random.lognormal(11, 0.8, n)
    deductions = np.clip(income * np.random.uniform(0.05, 0.6, n), 0, 200_000)
    credits = np.clip(income * np.random.uniform(0.01, 0.15, n), 0, 30_000)
    fraud_prob = (
        (deductions / income > 0.5).astype(int) * 0.6 +
        (credits / income > 0.1).astype(int) * 0.4 +
        np.random.rand(n) * 0.2
    )
    is_fraud = (fraud_prob > 0.65).astype(int)

    df = pd.DataFrame({"income": income, "deductions": deductions, "credits": credits, "is_fraud": is_fraud})
    X, y = df.drop("is_fraud", axis=1), df["is_fraud"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    return clf, df

model, full_df = train_model()

# ------------------ Sidebar Inputs ------------------
with st.sidebar:
    st.header("Tax Return Inputs")
    income = st.slider("Annual Income ($)", 10_000, 500_000, 75_000, step=5_000, format="$%d")
    ded_ratio = st.slider("Deduction % of Income", 0.0, 100.0, 20.0, step=1.0)
    deductions = income * ded_ratio / 100
    cred_ratio = st.slider("Credit % of Income", 0.0, 30.0, 4.0, step=0.5)
    credits = income * cred_ratio / 100

    st.markdown("---")
    if st.button("Run Fraud Check", type="primary", use_container_width=True):
        st.session_state.run = True
    else:
        st.session_state.run = False

# ------------------ Prediction & Risk Gauge ------------------
if st.session_state.run:
    prob = model.predict_proba([[income, deductions, credits]])[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    color = {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#10b981"}[risk]

    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Fraud Risk: {risk}</b>", 'font': {'size': 24}},
        delta={'reference': 30, 'position': "top"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#dcfce7"},
                {'range': [30, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#fecaca"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'>Income<br><b>${income:,.0f}</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'>Deductions<br><b>${deductions:,.0f}</b></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'>Credits<br><b>${credits:,.0f}</b></div>", unsafe_allow_html=True)

    # Alert
    if risk == "HIGH":
        st.error("High fraud likelihood – recommend full audit.")
    elif risk == "MEDIUM":
        st.warning("Moderate risk – review deduction sources.")
    else:
        st.success("Low risk – return appears clean.")

# ------------------ Interactive Data Explorer ------------------
with st.expander("Explore Training Data & Model Insights", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Fraud Distribution")
        fraud_counts = full_df["is_fraud"].value_counts()
        fig_pie = px.pie(values=fraud_counts.values, names=["Clean", "Fraud"], color_discrete_sequence=["#86efac", "#fca5a5"])
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("#### Income vs Deductions (Fraud Highlighted)")
        fig_scatter = px.scatter(
            full_df, x="income", y="deductions", color="is_fraud",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            labels={"is_fraud": "Fraud"}, opacity=0.7
        )
        fig_scatter.add_vline(x=income, line_dash="dash", line_color="orange")
        fig_scatter.add_hline(y=deductions, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature Importance
    perm_importance = permutation_importance(model, full_df.drop("is_fraud", axis=1), full_df["is_fraud"], n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({
        "Feature": ["Income", "Deductions", "Credits"],
        "Importance": perm_importance.importances_mean
    }).sort_values("Importance", ascending=False)

    fig_bar = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance",
                     color_continuous_scale="Viridis", title="Permutation Feature Importance")
    st.plotly_chart(fig_bar, use_container_width=True)

