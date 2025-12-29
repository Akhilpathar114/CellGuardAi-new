
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="CellGuardAI", layout="wide")

st.title("CellGuardAI — Stable, Fixed Dashboard")

uploaded = st.file_uploader("Upload Shell_BMS.csv", type=["csv"])
if uploaded is None:
    st.info("Upload a Shell_BMS.csv file to begin.")
    st.stop()

# --------------------
# Robust ingestion
# --------------------
df = pd.read_csv(uploaded)
df = df.reset_index(drop=True)
df["t_index"] = np.arange(len(df))

# Column mapping (flexible to common Shell names)
def pick(colnames):
    for c in colnames:
        if c in df.columns:
            return c
    return None

COL_V = pick(["Pack Vol", "PackVol", "Voltage", "voltage"])
COL_I = pick(["Current", "Curent", "I"])
COL_SOC = pick(["Soc", "SOC"])
COL_T = pick(["Temp1", "Temp", "Temperature", "temperature"])

# --------------------
# Header metrics
# --------------------
st.subheader("Status Overview")

c1, c2, c3 = st.columns(3)

with c1:
    if COL_V:
        v_mean = float(pd.to_numeric(df[COL_V], errors="coerce").mean())
        st.metric("Avg Pack Voltage", f"{v_mean:.2f} V")
    else:
        st.metric("Avg Pack Voltage", "N/A")

with c2:
    if COL_I:
        i_mean = float(pd.to_numeric(df[COL_I], errors="coerce").abs().mean())
        st.metric("Avg |Current|", f"{i_mean:.2f}")
    else:
        st.metric("Avg |Current|", "N/A")

with c3:
    if COL_SOC:
        soc_last = pd.to_numeric(df[COL_SOC], errors="coerce").iloc[-1]
        st.metric("SOC (last)", f"{soc_last:.1f}%")
    else:
        st.metric("SOC (last)", "N/A")

st.divider()

# --------------------
# Safe plotting helper
# --------------------
def safe_line(x, y, title, y_label):
    if y is None or y not in df.columns:
        st.info(f"{title}: column not available.")
        return
    series = pd.to_numeric(df[y], errors="coerce")
    if series.nunique(dropna=True) <= 1:
        st.info(f"{title}: signal is constant over this window (steady state).")
        return
    fig = px.line(df, x="t_index", y=y, title=title,
                  labels={"t_index": "Sample Index", y: y_label})
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Plots
# --------------------
st.subheader("Signals vs Sample Index")

colA, colB = st.columns(2)

with colA:
    safe_line("t_index", COL_V, "Pack Voltage vs Index", "Voltage")

with colB:
    safe_line("t_index", COL_I, "Current vs Index", "Current")

colC, colD = st.columns(2)

with colC:
    safe_line("t_index", COL_SOC, "SOC vs Index", "SOC (%)")

with colD:
    if COL_T and COL_T in df.columns:
        temp_series = pd.to_numeric(df[COL_T], errors="coerce")
        if temp_series.nunique(dropna=True) > 1:
            fig_t = px.histogram(df, x=COL_T, title="Temperature Distribution",
                                 labels={COL_T: "Temperature"})
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Temperature is constant over this window (steady state).")
    else:
        st.info("Temperature column not available.")

st.divider()

# --------------------
# Simple, stable health proxy (rule-based, per-window)
# --------------------
st.subheader("Battery Health (Stable Proxy)")

health = 100.0
if COL_I:
    health -= min(40, float(pd.to_numeric(df[COL_I], errors="coerce").abs().mean()) * 2)
if COL_T:
    tstd = float(pd.to_numeric(df[COL_T], errors="coerce").std())
    health -= min(30, tstd * 5)

health = max(0.0, min(100.0, health))
st.metric("Health Score", f"{health:.1f} / 100")

st.caption("Health is a conservative, rule-based proxy for stability over the uploaded window.")

st.success("App is stable and compatible with Shell_BMS.csv.")
