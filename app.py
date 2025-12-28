
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="CellGuardAI — Shell BMS", layout="wide")
st.title("🔋 CellGuardAI")
st.caption("Battery intelligence dashboard (Shell BMS compatible)")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Setup")
uploaded_file = st.sidebar.file_uploader("Upload Shell_BMS.csv", type=["csv"])

with st.sidebar.expander("Advanced controls (safe)"):
    contamination = st.slider("Anomaly sensitivity", 0.01, 0.1, 0.03)
    window = st.slider("Analysis window", 10, 100, 40)

# =====================================================
# SHELL COLUMN MAPPING (FIXED)
# =====================================================
def map_shell_csv(df):
    df = df.copy()

    # Time
    if "Date" in df.columns and "Time" in df.columns:
        df["time"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
    else:
        df["time"] = np.arange(len(df))

    # Pack signals
    df["voltage"] = pd.to_numeric(df.get("Pack Vol"), errors="coerce")
    df["current"] = pd.to_numeric(df.get("Curent"), errors="coerce")

    # Temperatures (avg)
    temp_cols = [c for c in df.columns if c.lower().startswith("temp")]
    df["temperature"] = df[temp_cols].astype(float).mean(axis=1) if temp_cols else np.nan

    # Cells (extra info, not core)
    cell_cols = [c for c in df.columns if c.lower().startswith("cell")]
    if cell_cols:
        df["cell_min"] = df[cell_cols].min(axis=1)
        df["cell_max"] = df[cell_cols].max(axis=1)
        df["cell_spread"] = df["cell_max"] - df["cell_min"]
        df["weakest_cell"] = df[cell_cols].idxmin(axis=1)
    else:
        df["cell_spread"] = np.nan
        df["weakest_cell"] = "N/A"

    df = df.dropna(subset=["voltage", "current"])
    df = df.sort_values("time").reset_index(drop=True)

    return df, cell_cols

# =====================================================
# ML (SIMPLE, STABLE)
# =====================================================
def compute_anomaly_risk(df, window, contamination):
    if len(df) <= window:
        return np.zeros(len(df))

    feats = []
    for i in range(window, len(df)):
        w = df.iloc[i-window:i]
        feats.append([
            w["voltage"].mean(),
            w["current"].mean(),
            w["temperature"].mean()
        ])

    X = StandardScaler().fit_transform(np.array(feats))

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X)

    score = -model.decision_function(X)
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)

    risk = np.zeros(len(df))
    risk[window:] = score
    return risk

# =====================================================
# MAIN
# =====================================================
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df, cell_cols = map_shell_csv(df_raw)

    risk = compute_anomaly_risk(df, window, contamination)
    health = max(0, 100 * (1 - risk[-1]))

    # =================================================
    # STATUS
    # =================================================
    st.markdown("## 🔋 Battery Status")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Health score", f"{health:.1f} / 100")

    with c2:
        if health > 70:
            st.success("Status: NORMAL")
        elif health > 45:
            st.warning("Status: WATCH")
        else:
            st.error("Status: CRITICAL")

    with c3:
        st.metric("Avg Temperature (°C)", f"{df['temperature'].iloc[-1]:.1f}")

    with c4:
        st.metric("Weakest Cell", df["weakest_cell"].iloc[-1])

    st.divider()

    # =================================================
    # TRENDS
    # =================================================
    st.markdown("## 📈 Trends")

    t1, t2 = st.columns(2)
    with t1:
        st.line_chart(df["voltage"])
        st.caption("Pack voltage")

    with t2:
        st.line_chart(df["current"])
        st.caption("Current")

    if "cell_spread" in df.columns:
        st.line_chart(df["cell_spread"])
        st.caption("Cell voltage spread (informational)")

    st.divider()

    # =================================================
    # ALERTS
    # =================================================
    st.markdown("## 🚨 Alerts")

    if health < 45:
        st.error("Abnormal operating behavior detected. Inspect pack.")
    elif health < 70:
        st.warning("Minor anomalies observed. Monitor closely.")
    else:
        st.success("Battery operating normally.")

    # =================================================
    # CELLS (EXTRA VALUE)
    # =================================================
    if cell_cols:
        st.markdown("## 🧬 Cell Snapshot")
        cell_df = pd.DataFrame({
            "Cell": cell_cols,
            "Voltage": df[cell_cols].iloc[-1].values
        }).sort_values("Voltage")
        st.dataframe(cell_df, use_container_width=True)

    # =================================================
    # RAW DATA
    # =================================================
    with st.expander("Raw mapped data preview"):
        st.dataframe(df.head(50))

else:
    st.info("Upload Shell_BMS.csv to start analysis")
