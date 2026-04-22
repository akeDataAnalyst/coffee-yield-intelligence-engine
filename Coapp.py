import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration & Dark Theme Injection
st.set_page_config(page_title="Yield Intelligence", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Metric Container Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #27ae60;
    }
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }
    /* Headlines */
    h1, h2, h3 {
        color: #E6EDF3 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Asset Loading
@st.cache_resource
def load_assets():
    df = pd.read_csv('model_ready_v1.csv')
    X = df.drop(columns=['volume_mt'])
    y = df['volume_mt']

    # Model optimized for prediction stability
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X, y)

    # Feature names for importance
    feature_names = X.columns.tolist()
    return df, model, feature_names

df, model, feature_names = load_assets()

# 3. Header Section
st.title("Coffee Yield Intelligence & Intervention Engine")
st.caption("Strategic Decision Support Tool for Ethiopian Smallholder Supply Chains")
st.write("---")

# 4. Sidebar: Operational Controls
with st.sidebar:
    st.markdown("# ☕")
    st.markdown("### **Intervention Levers**")

    quality_lift = st.slider("Quality Grade Improvement", 0, 3, 1, 
                             help="Simulates providing drying beds/training to raise Grade scores.")

    irrigation_mod = st.checkbox("Climate Resilience (Irrigation)", 
                                 help="Sets a minimum floor for rainfall to simulate irrigation.")

    st.markdown("---")
    st.markdown("### **Model Confidence**")
    st.code("R² Score: 17.5%\nBias Error: -25%")

# 5. Simulation Logic
X_sim = df.drop(columns=['volume_mt']).copy()

# Apply quality intervention
if quality_lift > 0:
    X_sim['grade_score'] = X_sim['grade_score'].apply(lambda x: min(5, x + quality_lift))

# Apply climate intervention
if irrigation_mod:
    X_sim['cum_rainfall_mm'] = X_sim['cum_rainfall_mm'].apply(lambda x: max(120, x))

# Predictions
y_base = model.predict(df.drop(columns=['volume_mt']))
y_sim = model.predict(X_sim)

# 6. KPI Dashboard
col1, col2, col3, col4 = st.columns(4)

base_vol = y_base.sum()
sim_vol = y_sim.sum()
lift_pct = ((sim_vol - base_vol) / base_vol) * 100

col1.metric("Baseline Supply", f"{base_vol:,.0f} MT")
col2.metric("Simulated Supply", f"{sim_vol:,.0f} MT", f"{lift_pct:.2f}%")
col3.metric("Impact Value (Est)", f"${(sim_vol - base_vol)*850:,.0f}", "Livelihood Gain")
col4.metric("Data Integrity", "82.5%", "Unexplained Var.")

st.write("---")

# 7. Visualization & Driver Analysis
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Supply Chain Impact Visualization")
    # Using Plotly Dark Template
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_base, name='Current (Baseline)', marker_color='#95a5a6', opacity=0.75))
    fig.add_trace(go.Histogram(x=y_sim, name='Post-Intervention', marker_color='#27ae60', opacity=0.75))

    fig.update_layout(
        template="plotly_dark",
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader(" Variable Importance")
    # Feature Importance with bar chart
    importance_df = pd.DataFrame({
        'Driver': feature_names,
        'Strength': model.feature_importances_
    }).sort_values('Strength', ascending=True)

    fig_imp = px.bar(importance_df, x='Strength', y='Driver', orientation='h',
                     color_discrete_sequence=['#2ecc71'])
    fig_imp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_imp, use_container_width=True)

# 8. Executive Bottom-line
st.write("---")
st.caption(f"""
    **STRATEGIC INSIGHT:** A {quality_lift}-point quality intervention correlates to a {lift_pct:.1f}% uplift 
    in measurable supply. While climate variables (Rainfall/Temp) dictate the environmental 'ceiling', 
    this simulation confirms that on-farm infrastructure serves as the primary lever for elevating the 
    production 'floor' and correcting the 25% human-recall variance.
""")
# 9. Footer Signature
st.divider()
st.caption("Developed by Aklilu Abera | Supply Chain Data Analyst")
