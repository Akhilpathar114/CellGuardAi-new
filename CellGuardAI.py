
# CellGuardAI.py (BUG-FIX-ONLY PATCH)
# UI unchanged. Fixes:
# - guard missing columns in plots
# - index-based time axis
# - hide debug tables
# - prevent Plotly ValueError on constant signals

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def safe_line(df, x, y, title):
    if y not in df.columns:
        st.info(f"{title}: data not available")
        return
    if df[y].nunique() <= 1:
        st.info(f"{title}: signal is constant in this dataset")
        return
    fig = px.line(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("CELLGUARD.AI — Dashboard")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        st.stop()

    df = pd.read_csv(uploaded)
    df['t_index'] = np.arange(len(df))

    # --- UI BELOW IS UNCHANGED IN STRUCTURE ---
    st.subheader("AI-Based Battery Insights")

    safe_line(df, 't_index', 'battery_health_score', "Health Score Over Time")
    safe_line(df, 't_index', 'Pack Vol', "Pack Voltage")
    safe_line(df, 't_index', 'Soc', "SOC Trend")
    safe_line(df, 't_index', 'Current', "Current Flow Over Time")

if __name__ == "__main__":
    main()
