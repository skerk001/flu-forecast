"""
Streamlit dashboard for the flu forecast.

Run locally:
    streamlit run dashboard/app.py

Or deploy free at https://streamlit.io/cloud — point it at this repo.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Flu Forecast", layout="wide",
                   page_icon="🤒")

st.title("🤒 CDC FluView ILI Forecast")
st.caption("Weekly influenza-like illness forecast for the United States. "
           "Data: CDC FluView (ILINet) via the Delphi Epidata API.")

# Load data
@st.cache_data(ttl=3600)
def load_history():
    return pd.read_csv(ROOT / "data" / "fluview_national.csv", parse_dates=["date"])

@st.cache_data(ttl=3600)
def load_forecasts():
    out = {}
    for f in (ROOT / "outputs" / "forecasts").glob("*_forecast.csv"):
        name = f.stem.replace("_forecast", "").upper()
        out[name] = pd.read_csv(f, index_col=0, parse_dates=True)
    return out

@st.cache_data(ttl=3600)
def load_run_meta():
    p = ROOT / "outputs" / "latest_run.json"
    return json.loads(p.read_text()) if p.exists() else {}


df = load_history()
forecasts = load_forecasts()
meta = load_run_meta()

# Sidebar controls
st.sidebar.header("Controls")
years_back = st.sidebar.slider("Years of history to show", 1, 15, 5)
selected_models = st.sidebar.multiselect(
    "Models to display", list(forecasts.keys()), default=list(forecasts.keys()),
)

# Top-level KPIs
latest_obs = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest week", latest_obs["date"].strftime("%Y-%m-%d"))
col2.metric("Latest %ILI", f"{latest_obs['wili']:.2f}")
col3.metric("# Providers", f"{int(latest_obs['num_providers']):,}")
col4.metric("Patient visits", f"{int(latest_obs['num_patients']):,}")

if meta.get("metrics"):
    st.subheader("Model performance (walk-forward CV)")
    st.dataframe(pd.DataFrame(meta["metrics"]).set_index("model"),
                 use_container_width=True)

# Forecast chart
st.subheader("Forecast")
cutoff = df["date"].max() - pd.DateOffset(years=years_back)
hist = df[df["date"] >= cutoff]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["wili"], name="Actual %ILI",
                         line=dict(color="black", width=2)))
colors = {"SARIMA": "#d62728", "PROPHET": "#1f77b4", "LSTM": "#2ca02c"}
for name in selected_models:
    fc = forecasts[name]
    c = colors.get(name, "#888")
    fig.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["upper"]) + list(fc["lower"])[::-1],
        fill="toself", fillcolor=c, opacity=0.15, line=dict(width=0),
        name=f"{name} 95% CI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name=f"{name} forecast",
                             line=dict(color=c, width=3)))

fig.update_layout(hovermode="x unified", height=500,
                  xaxis_title="Date", yaxis_title="% ILI visits",
                  template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Decomposition image
st.subheader("Seasonal-trend decomposition")
decomp_path = ROOT / "outputs" / "plots" / "decomposition.png"
if decomp_path.exists():
    st.image(str(decomp_path), use_container_width=True)

# Raw data
with st.expander("Show raw history"):
    st.dataframe(df.tail(52), use_container_width=True)

st.caption(f"Last pipeline run: {meta.get('run_time_utc', 'unknown')}")
