# frontend/pages/2_TimeSeries.py
import sys
import os

# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import csv
from app.timeseries_engine import train_and_forecast


st.set_page_config(page_title="Time Series Forecasting", page_icon="📈", layout="wide")
st.title("📈 Time Series Forecasting Dashboard")

# === 0. Lecture intelligente du séparateur ===
def smart_read_csv(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    uploaded_file.seek(0)  # remettre le curseur à zéro après read()

    # Détection automatique du séparateur
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(content.splitlines()[0])
    sep = dialect.delimiter

    df = pd.read_csv(uploaded_file, sep=sep)
    return df

# === 1. Upload CSV ===
st.markdown("### 📁 Upload your time series dataset")
uploaded_file = st.file_uploader("Upload a CSV file with a time column and a value column", type="csv")

if uploaded_file:
    df = smart_read_csv(uploaded_file)

    # === 2. Select Columns ===
    st.markdown("### 🧭 Select Columns")
    col1, col2 = st.columns(2)
    with col1:
        time_col = st.selectbox("Time Column", df.columns, index=0)
    with col2:
        value_col = st.selectbox("Value Column", df.columns, index=1)

    # Create a time index for modeling
    df = df[[time_col, value_col]].dropna()
    df["time_idx"] = range(len(df))

    # === 3. Choose model ===
    st.markdown("### ⚙️ Select Forecasting Model")
    model_type = st.selectbox("📊 Model", ["baseline", "linear", "random_forest", "xgboost", "arima"])

    # === 4. Run Forecasting ===
    st.markdown("### 🔮 Forecast Result")
    df_pred, mae, mape = train_and_forecast(df, x_col="time_idx", y_col=value_col, model_type=model_type)

    # === 5. Plot Graph ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df_pred["Actual"], mode="lines+markers", name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df_pred["Predicted"], mode="lines+markers", name="Forecast"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # === 6. Metrics ===
    st.markdown("### 📌 Forecast Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", f"{mae:.2f}")
    with col2:
        st.metric("MAPE", f"{mape:.2f} %")

else:
    st.info("👈 Upload a CSV file with a time and value column to get started.")
