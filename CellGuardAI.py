import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import re

st.set_page_config(page_title="CellGuard.AI - Dashboard", layout="wide")

# ================= PHASE 1: INTERPRETABLE FEATURES =================

def detect_operating_regime(df):
    df = df.copy()
    I = df["current"].abs()
    I_idle = I.quantile(0.05)
    I_stress = I.quantile(0.90)

    regimes = []
    for i in range(len(df)):
        cur = df.loc[i, "current"]
        if abs(cur) <= I_idle:
            regimes.append("Idle")
        elif cur > 0:
            regimes.append("Charging")
        else:
            regimes.append("Discharging")

    df["regime"] = regimes

    if "temperature" in df.columns:
        temp_rise = df["temperature"].diff().fillna(0)
        stress_mask = (I >= I_stress) & (temp_rise > 0)
        df.loc[stress_mask, "regime"] = "High-Stress"

    return df


def compute_stability_score(df, window=50):
    dv = df["voltage"].diff().rolling(window).std()
    di = df["current"].diff().rolling(window).std()
    dt = df["temperature"].diff().rolling(window).std()

    raw = dv + di + dt
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    stability = 100 * (1 - norm)

    return stability.fillna(method="bfill").clip(0, 100)


def weakest_cell_stats(df):
    cell_cols = [c for c in df.columns if c.lower().startswith("cell")]
    if not cell_cols:
        return None, None

    weakest_series = df[cell_cols].idxmin(axis=1)
    freq = weakest_series.value_counts(normalize=True) * 100

    return weakest_series.iloc[-1], freq.round(1)


# ================= PHASE 2: DIAGNOSTICS & CONFIDENCE =================

def compute_root_cause(df, window=50):
    dv = df["voltage"].diff().rolling(window).std()
    di = df["current"].diff().rolling(window).std()
    dt = df["temperature"].diff().rolling(window).std()

    latest = {
        "Voltage fluctuation": dv.iloc[-1],
        "Current fluctuation": di.iloc[-1],
        "Thermal fluctuation": dt.iloc[-1],
    }

    total = sum(v for v in latest.values() if pd.notna(v))
    if total == 0 or pd.isna(total):
        return None

    return {k: round(100 * v / total, 1) for k, v in latest.items()}


def compute_confidence_score(df):
    completeness = df.notna().mean().mean()
    variance_penalty = (
        df[["voltage", "current", "temperature"]]
        .diff()
        .std()
        .mean()
    )

    score = 100 * completeness / (1 + variance_penalty)
    return max(0, min(100, score))


def scenario_replay(df, idx):
    return df.iloc[: idx + 1]


# ================= PHASE 3: DECISION & LIFECYCLE =================

def estimate_rul_proxy(df, stability, confidence):
    if confidence < 40:
        return "Unknown (low confidence)"
    if stability > 80:
        return "Long remaining life"
    elif stability > 60:
        return "Moderate remaining life"
    elif stability > 40:
        return "Short remaining life"
    else:
        return "Immediate attention required"


def build_pack_summary(df, health, stability, confidence, rul_band):
    return {
        "Health Score": round(health, 1),
        "Stability Index": round(stability, 1),
        "Confidence": round(confidence, 1),
        "RUL Band": rul_band,
        "Weakest Cell": df.get("weakest_cell", pd.Series(["N/A"])).iloc[-1],
        "Dominant Regime": df["regime"].value_counts().idxmax()
    }


def export_snapshot(summary):
    return pd.DataFrame([summary])


# ----------------------
# Simple data generator
# ----------------------
def gen_sample_data(n=800, seed=42, scenario="Generic"):
    # basic simulated BMS-like data, a bit noisy
    np.random.seed(seed)
    t = np.arange(n)
    base_v = 3.7
    base_i = 1.5
    base_temp = 30.0
    soc_base = 80.0

    if scenario == "Generic":
        v = base_v + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
        i = base_i + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
        temp = base_temp + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
        soc = np.clip(soc_base + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
        cycle = t // 50
        # inject some faults
        idx = np.random.choice(n, size=18, replace=False)
        v[idx] -= np.random.uniform(0.03, 0.08, size=len(idx))
        temp[idx] += np.random.uniform(2, 5, size=len(idx))

    elif scenario == "EV":
        v = base_v + 0.03 * np.sin(t / 40) - 0.0005 * t / n + np.random.normal(0, 0.008, n)
        i = 2.5 + 0.4 * np.sin(t / 20) + np.random.normal(0, 0.07, n)
        temp = base_temp + 4 * np.sin(t / 120) + 0.01 * (t / n) * 10 + np.random.normal(0, 0.5, n)
        soc = np.clip(90 - 20 * (t / n) + np.random.normal(0, 1.5, n), 0, 100)
        cycle = t // 10
        idx = np.random.choice(n, size=35, replace=False)
        v[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
        temp[idx] += np.random.uniform(3, 8, size=len(idx))

    elif scenario == "Drone":
        v = base_v + 0.04 * np.sin(t / 30) + np.random.normal(0, 0.006, n)
        i = base_i + 0.6 * np.sin(t / 10) + np.random.normal(0, 0.2, n)
        temp = base_temp + 2 * np.sin(t / 80) + np.random.normal(0, 0.4, n)
        soc = np.clip(85 + 6 * np.sin(t / 40) + np.random.normal(0, 2, n), 0, 100)
        cycle = t // 30
        spikes = np.random.choice(n, size=60, replace=False)
        i[spikes] += np.random.uniform(2.0, 6.0, size=len(spikes))
        dips = np.random.choice(n, size=30, replace=False)
        v[dips] -= np.random.uniform(0.06, 0.18, size=len(dips))

    elif scenario == "Phone":
        v = base_v + 0.02 * np.sin(t / 80) + np.random.normal(0, 0.002, n)
        i = 0.8 + 0.1 * np.sin(t / 60) + np.random.normal(0, 0.02, n)
        temp = base_temp + 1.5 * np.sin(t / 120) + np.random.normal(0, 0.15, n)
        soc = np.clip(95 + 3 * np.sin(t / 160) + np.random.normal(0, 0.5, n), 0, 100)
        cycle = t // 200
        idx = np.random.choice(n, size=6, replace=False)
        v[idx] -= np.random.uniform(0.01, 0.03, size=len(idx))

    else:
        return gen_sample_data(n=n, seed=seed, scenario="Generic")

    df = pd.DataFrame({
        "time": t,
        "voltage": v,
        "current": i,
        "temperature": temp,
        "soc": soc,
        "cycle": cycle
    })
    return df


# ----------------------
# Helpers: rename/care for columns
# ----------------------
def normalize_cols(df):
    # try to map different column namings to our standard names
    df = df.copy()
    simple = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in df.columns}
    patt = {
        "voltage": ["volt", "vcell", "cellv", "packv"],
        "current": ["curr", "amp", "amps", "ichg", "idis", "current"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge"],
        "cycle": ["cycle", "cyclecount", "chargecycle"],
        "time": ["time", "timestamp", "t", "index"]
    }
    cmap = {}
    used = set()
    for target, keys in patt.items():
        for orig, simplified in simple.items():
            if orig in used:
                continue
            if any(k in simplified for k in keys):
                cmap[target] = orig
                used.add(orig)
                break
    rename = {orig: targ for targ, orig in cmap.items()}
    df = df.rename(columns=rename)
    return df, cmap


def ensure_cols_exist(df, needed):
    # add missing columns with NaN
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ----------------------
# Simple feature engineering
# ----------------------
def make_features(df, window=10):
    df = df.copy()
    df = ensure_cols_exist(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    # voltage features
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan

    # temperature features
    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan

    # soc features
    if df["soc"].notna().sum() > 0:
        df["soc_ma"] = df["soc"].rolling(window, min_periods=1).mean()
        df["soc_roc"] = df["soc"].diff().fillna(0)
    else:
        df["soc_ma"] = np.nan
        df["soc_roc"] = np.nan

    # risk label: simple rules (voltage drop, temp spike, soc drop)
    if df["voltage"].notna().sum() > 0:
        volt_drop_thresh = -0.03
        cond = pd.Series(False, index=df.index)
        if df["temperature"].notna().sum() > 0:
            tmean = df["temperature"].mean()
            tstd = df["temperature"].std()
            tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
            if not np.isnan(tth):
                cond = cond | (df["temperature"] > tth)
        if "voltage_roc" in df.columns:
            cond = cond | (df["voltage_roc"] < volt_drop_thresh)
        if "soc_roc" in df.columns:
            cond = cond | (df["soc_roc"] < -5)
        df["risk_label"] = np.where(cond, 1, 0)
    else:
        df["risk_label"] = 0

    return df


# ----------------------
# Models + scoring
# ----------------------
def run_models(df, contamination=0.05):
    df = df.copy()
    # candidate features for anomaly detection
    possible = ["voltage", "current", "temperature", "soc", "voltage_ma", "voltage_roc", "soc_roc", "voltage_var", "temp_ma", "cycle"]
    features = [f for f in possible if f in df.columns and df[f].notna().sum() > 0]

    df["anomaly_flag"] = 0
    df["risk_pred"] = 0
    df["battery_health_score"] = 50.0

    # Isolation Forest for anomalies if enough data
    if len(features) >= 2 and df[features].dropna().shape[0] >= 30:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[features].fillna(df[features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            df["anomaly_flag"] = 0

    # If risk_label exists, train a small decision tree to mimic it
    if "risk_label" in df.columns and df["risk_label"].nunique() > 1:
        clf_feats = [f for f in features if f in df.columns]
        if len(clf_feats) >= 2:
            try:
                Xc = df[clf_feats].fillna(df[clf_feats].median())
                yc = df["risk_label"]
                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(Xc, yc)
                df["risk_pred"] = tree.predict(Xc)
            except Exception:
                df["risk_pred"] = df["risk_label"]
        else:
            df["risk_pred"] = df["risk_label"]
    else:
        # fallback: use temperature threshold + anomalies
        tseries = df.get("temperature", pd.Series(np.nan, index=df.index))
        tmean = tseries.mean() if hasattr(tseries, "mean") else np.nan
        tstd = tseries.std() if hasattr(tseries, "std") else np.nan
        tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
        cond_temp = (tseries > tth) if not np.isnan(tth) else False
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | cond_temp, 1, 0)

    # Build a base score vector from voltage and temp and anomaly/risk
    base = pd.Series(0.0, index=df.index)
    if "voltage_ma" in df.columns and df["voltage_ma"].notna().sum() > 0:
        vm = df["voltage_ma"].fillna(method="ffill").fillna(df["voltage"].median() if "voltage" in df.columns else 3.7)
        base += (vm.max() - vm)
    elif "voltage" in df.columns:
        v = df["voltage"].fillna(df["voltage"].median())
        base += (v.max() - v)
    else:
        base += 0.5

    if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
        t = df["temperature"].fillna(df["temperature"].median())
        base += (t - t.min()) / 10.0

    base = base + df.get("anomaly_flag", 0)*1.0 + df.get("risk_pred", 0)*0.8

    # try to fit a linear reg on some trend features to get hp-like value
    trend_feats = [f for f in ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"] if f in df.columns]
    if len(trend_feats) >= 2 and df[trend_feats].dropna().shape[0] >= 20:
        try:
            Xtr = df[trend_feats].fillna(0)
            reg = LinearRegression()
            reg.fit(Xtr, base)
            hp = reg.predict(Xtr)
        except Exception:
            hp = base.values
    else:
        hp = base.values

    hp = np.array(hp, dtype=float)
    hp_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)
    health_comp = 1 - hp_norm

    score = (0.6 * health_comp) + (0.25 * (1 - df.get("risk_pred", 0))) + (0.15 * (1 - df.get("anomaly_flag", 0)))
    df["battery_health_score"] = (score * 100).clip(0, 100)

    return df


# ----------------------
# Simple recommendations & labels
# ----------------------
def simple_recommend(row):
    sc = row.get("battery_health_score", 50)
    rp = row.get("risk_pred", 0)
    an = row.get("anomaly_flag", 0)
    if sc > 85 and rp == 0 and an == 0:
        return "Healthy — normal operation."
    elif 70 < sc <= 85:
        return "Watch — avoid deep discharge & fast-charge this cycle."
    elif 50 < sc <= 70:
        return "Caution — restrict fast charging; allow cooling intervals."
    else:
        return "Critical — reduce load, stop fast charging, schedule inspection."


def pack_label(score):
    if score >= 85:
        return "HEALTHY", "green"
    elif score >= 60:
        return "WATCH", "orange"
    else:
        return "CRITICAL", "red"


def make_gauge_figure(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Battery Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightcoral"},
                {'range': [60, 85], 'color': "gold"},
                {'range': [85, 100], 'color': "lightgreen"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
    return fig


def anomaly_markers(df):
    adf = df[df.get("anomaly_flag", 0) == 1]
    if adf.empty:
        return None
    return go.Scatter(x=adf["time"], y=adf["battery_health_score"], mode="markers", name="Anomaly", marker=dict(color="red", size=8, symbol="x"))


def basic_alerts(df):
    alerts = []
    if "temperature" in df.columns and df["temperature"].notna().sum()>0:
        tmean = df["temperature"].mean()
        tstd = df["temperature"].std()
        recent = df["temperature"].iloc[-1]
        if recent > (tmean + 2*tstd):
            alerts.append({"title":"Thermal drift", "detail":"Temp well above normal — hotspot risk. Cool & inspect.", "severity":"high"})
    if "voltage_roc" in df.columns and "voltage_var" in df.columns:
        last_roc = df["voltage_roc"].rolling(5).mean().iloc[-1]
        last_var = df["voltage_var"].rolling(10).mean().iloc[-1]
        if last_roc < -0.01:
            alerts.append({"title":"Voltage sag pattern", "detail":"Sustained negative voltage change — internal resistance rising.", "severity":"medium"})
        if last_var > df["voltage_var"].mean() + df["voltage_var"].std():
            alerts.append({"title":"Voltage variance rising", "detail":"Cell-to-cell variance increasing — imbalance risk.", "severity":"medium"})
    if "current" in df.columns and df["current"].notna().sum()>0:
        spike_pct = (df["current"] > (df["current"].mean() + 2*df["current"].std())).mean()
        if spike_pct > 0.02:
            alerts.append({"title":"Current spikes", "detail":"Frequent high current spikes — mechanical/connection stress likely.", "severity":"medium"})
    if "anomaly_flag" in df.columns:
        p = df["anomaly_flag"].mean()
        if p > 0.05:
            alerts.append({"title":"Anomaly rate high", "detail":f"{p*100:.1f}% readings flagged — investigate.", "severity":"medium"})
    if "risk_pred" in df.columns and df["risk_pred"].iloc[-1]==1:
        alerts.append({"title":"Immediate risk", "detail":"Model predicts elevated risk on latest measurement.", "severity":"high"})
    return alerts


def top_recs_from_df(df, n=5):
    out = []
    try:
        if "battery_health_score" in df.columns:
            worst = df.nsmallest(n, "battery_health_score")
            if "recommendation" in worst.columns:
                rec_counts = worst["recommendation"].value_counts()
                for rec, cnt in rec_counts.items():
                    out.append({"recommendation": rec, "count": int(cnt)})
    except Exception:
        pass
    return out


# ----------------------
# Export PDF
# ----------------------
def make_pdf(df_out, avg_score, anomaly_pct, alerts, recs, verdict_text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 18*mm
    x = margin
    y = h - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "CellGuard.AI — Diagnostic Report")
    y -= 8*mm

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Avg Health Score: {avg_score:.1f}/100")
    c.drawString(x + 80*mm, y, f"Anomaly Rate: {anomaly_pct:.2f}%")
    y -= 6*mm
    c.drawString(x, y, f"Data points: {len(df_out)}")
    y -= 8*mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Combined Verdict:")
    c.setFont("Helvetica", 11)
    c.drawString(x + 30, y, verdict_text)
    y -= 10*mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top AI Alerts:")
    y -= 6*mm
    c.setFont("Helvetica", 10)
    if alerts:
        for a in alerts[:6]:
            c.drawString(x + 6, y, f"- {a['title']}: {a['detail']}")
            y -= 5*mm
            if y < margin + 40*mm:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6*mm

    y -= 4*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top Recommendations:")
    y -= 6*mm
    c.setFont("Helvetica", 10)
    if recs:
        for r in recs[:6]:
            c.drawString(x + 6, y, f"- {r['recommendation']} (observed {r['count']} times)")
            y -= 5*mm
            if y < margin + 20*mm:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6*mm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin, "Generated by CellGuard.AI")
    c.save()
    buf.seek(0)
    return buf.read()


# safe chart display (small wrapper)
def safe_plot(fig, key, **kwargs):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key, **kwargs)
    except Exception as e:
        st.warning(f"Chart '{key}' failed: {e}")


# ----------------------
# CSV sanitizer for uploaded files
# ----------------------
def strong_sanitize(df):
    # This is a pretty strong cleaning function to handle weird CSVs from judges or test sets.
    df = df.copy()
    st.write("### Raw sample (first 6 rows)")
    st.write(df.head(6))
    st.write("### Raw dtypes")
    st.write(df.dtypes)

    # normalize col names
    df.columns = [str(c).strip() for c in df.columns]

    # try to drop repeated header rows (some CSVs have headers repeated)
    def looks_like_header(row):
        cnt = 0
        for col in df.columns:
            try:
                if str(row.get(col, "")).strip().lower() == str(col).strip().lower():
                    cnt += 1
            except Exception:
                pass
        return cnt >= max(2, len(df.columns)//4)

    header_mask = df.apply(looks_like_header, axis=1)
    if header_mask.any():
        st.warning(f"Removed {header_mask.sum()} header-like rows.")
        df = df.loc[~header_mask].reset_index(drop=True)

    force_numeric = ["voltage","current","temperature","temp","soc","cycle","time"]
    lower_map = {c.lower(): c for c in df.columns}
    mapped = {}
    for want in force_numeric:
        for clower, orig in lower_map.items():
            if want in clower or (any(k in clower for k in ["volt","vcell","packv"]) and "volt" in want):
                if want not in mapped:
                    mapped[want] = orig

    for core in ["voltage","current","temperature","soc","cycle","time"]:
        if core in df.columns and core not in mapped:
            mapped[core] = core

    st.write("Column mapping (for force-numeric):", mapped)

    def clean_num_str(s):
        if pd.isnull(s):
            return s
        s = str(s).strip()
        s = re.sub(r'(?i)\b(v|volts?|a|amps?|%|degc|c|khz|hz|mv|ma)\b', '', s)
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^\d\.\-eE,]+', '', s)
        s = s.replace(',', '')
        s = s.strip()
        return s if s != '' else None

    for key, orig_col in mapped.items():
        try:
            raw = df[orig_col].astype(object)
        except Exception:
            continue

        cleaned = raw.map(clean_num_str)
        coerced = pd.to_numeric(cleaned, errors='coerce')

        bad_mask = coerced.isna() & raw.notna()
        if bad_mask.any():
            bad_vals = raw[bad_mask].astype(str).unique()[:50]
            st.write(f"Non-numeric examples in `{orig_col}` (up to 50):")
            st.write(list(bad_vals))

        df[orig_col] = coerced

        if df[orig_col].notna().sum() > 0:
            med = df[orig_col].median(skipna=True)
            df[orig_col] = df[orig_col].fillna(med).ffill().bfill()
        else:
            df[orig_col] = df[orig_col].astype(float)

    st.write("### After force-conversion dtypes")
    st.write(df.dtypes)
    st.write("### Cleaned sample (first 6 rows)")
    st.write(df.head(6))
    return df


# ----------------------
# Main app UI
# ----------------------
def main():
    st.title("CELLGUARD.AI — Dashboard")
    st.write("Predictive battery intelligence: health score, early alerts, anomaly timeline, and recommendations.")

    # sidebar
    st.sidebar.header("Config")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
    scenario = st.sidebar.selectbox("Demo scenario (if Sample data)", ["Generic", "EV", "Drone", "Phone"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    st.sidebar.markdown("Tip: upload CSV with columns like voltage, temperature, current, soc, time.")

    # load data
    if data_mode == "Sample data":
        df_raw = gen_sample_data(n=800, seed=42, scenario=scenario)
        st.sidebar.success(f"Using simulated data: {scenario}")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV or choose Sample data.")
            st.stop()
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception:
            try:
                df_raw = pd.read_csv(uploaded, encoding="latin1")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
        st.sidebar.success("CSV loaded.")
        df_raw = strong_sanitize(df_raw)

    # normalize + features + models
    df_raw, col_map = normalize_cols(df_raw)
    df_raw = ensure_cols_exist(df_raw, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    df_feat = make_features(df_raw, window=window)
    df_out = run_models(df_feat, contamination=contamination)
    df_out["recommendation"] = df_out.apply(simple_recommend, axis=1)

    avg_score = float(df_out["battery_health_score"].mean()) if not df_out["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_out.columns else 0.0
    label, color = pack_label(avg_score)

    alerts = basic_alerts(df_out)
    recs = top_recs_from_df(df_out, n=8)
    pdf_bytes = make_pdf(df_out, avg_score, anomaly_pct, alerts, recs, "Auto-generated verdict")

    # header columns
    left, mid, right = st.columns([1.4, 1.4, 1])
    with left:
        st.markdown("### Battery Health")
        gauge = make_gauge_figure(avg_score)
        safe_plot(gauge, key="gauge_health")
    with mid:
        st.markdown("### Pack Status")
        badge_color = "#2ecc71" if label=="HEALTHY" else ("#f39c12" if label=="WATCH" else "#e74c3c")
        st.markdown(f"<span style='background:{badge_color};color:#fff;padding:6px 10px;border-radius:8px;font-weight:600'>{label}</span>", unsafe_allow_html=True)
        st.metric("Avg Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f} vs ideal")
        st.write(f"- Scenario: **{scenario}**")
        st.write(f"- Anomalies: **{anomaly_pct:.1f}%**")
        st.write(f"- Data points: **{len(df_out)}**")
        st.write(f"- Mapped columns: {', '.join(list(col_map.keys())) if col_map else 'auto-map not found'}")
    with right:
        st.markdown("### Actions")
        st.download_button("⬇️ Download processed CSV", df_out.to_csv(index=False).encode("utf-8"),
                           "CellGuardAI_Output.csv", "text/csv", key="download_processed_csv_header")

    # verdict + downloads
    st.subheader("Combined Verdict and PDF")
    st.write("### Final Verdict")
    if avg_score < 60:
        st.error("Combined verdict: Immediate action required.")
    elif avg_score < 75:
        st.warning("Combined verdict: Monitor closely.")
    else:
        st.success("Combined verdict: Pack is healthy.")

    st.download_button("⬇️ Download Processed CSV", df_out.to_csv(index=False).encode("utf-8"),
                       file_name="CellGuardAI_Processed.csv", mime="text/csv", key="download_processed_csv")
    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="CellGuardAI_Report.pdf", mime="application/pdf", key="download_pdf_report")
    st.download_button("⬇️ Download Full Raw CSV", data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="CellGuardAI_FullOutput.csv", mime="text/csv", key="download_full_csv")

    st.markdown("### Predictive Alerts (main)")
    if alerts:
        for a in alerts:
            sev = a.get("severity", "info")
            if sev == "high":
                st.markdown(f"<div style='background:#fdecea;padding:8px;border-radius:8px;margin-bottom:6px'><b>🔴 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
            elif sev == "medium":
                st.markdown(f"<div style='background:#fff4e5;padding:8px;border-radius:8px;margin-bottom:6px'><b>🟠 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#eef7ff;padding:8px;border-radius:8px;margin-bottom:6px'><b>🔵 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
    else:
        st.success("No immediate AI alerts")

    st.markdown("---")

    # recommendations area
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Recommendations (if any)")
        if recs:
            for r in recs:
                st.markdown(f"✅ **{r['recommendation']}** — seen in ({r['count']}) risky rows")
        else:
            st.write("No specific recommendations at this time.")
    with c2:
        st.subheader("Top Warnings Snapshot")
        if alerts:
            for a in alerts:
                st.markdown(f"- **{a['title']}** — {a['detail']}")
        else:
            st.write("No warnings.")

    st.markdown("---")

    # summary metrics
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Avg Temp (°C)", f"{df_out['temperature'].mean():.2f}" if df_out['temperature'].notna().sum()>0 else "N/A")
    with s2:
        st.metric("Voltage Var (mean)", f"{df_out['voltage_var'].mean():.4f}" if "voltage_var" in df_out.columns else "N/A")
    with s3:
        st.metric("Cycle Count (max)", f"{int(df_out['cycle'].max())}" if df_out['cycle'].notna().sum()>0 else "N/A")
    with s4:
        st.metric("Anomaly %", f"{anomaly_pct:.2f}%")
    with s5:
        st.metric("Last Risk Pred", "HIGH" if df_out["risk_pred"].iloc[-1]==1 else "NORMAL")

    st.markdown("---")

    # tabs for charts and tables
    tab_ai, tab_trad, tab_compare, tab_table = st.tabs(["CellGuard.AI", "Traditional BMS", "Compare (Combined)", "Data"])

    with tab_ai:
        st.subheader("AI-Based Battery Insights")
        fig_h = px.line(df_out, x="time", y="battery_health_score", labels={"time":"Time","battery_health_score":"Health Score"}, title="Health Score Over Time")
        safe_plot(fig_h, key="ai_health_timeline")
        fig_vv = px.line(df_out, x="time", y="voltage_var", labels={"time":"Time","voltage_var":"Voltage Variance"}, title="Voltage Variance")
        safe_plot(fig_vv, key="ai_voltage_var")
        fig_soc_ai = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"SOC (%)"}, title="SOC Trend")
        safe_plot(fig_soc_ai, key="ai_soc_chart")
        fig_cur = px.line(df_out, x="time", y="current", labels={"time":"Time","current":"Current (A)"}, title="Current Flow Over Time")
        safe_plot(fig_cur, key="ai_current_plot")
        fig_temp_hist = px.histogram(df_out, x="temperature", labels={"temperature":"Temperature (°C)"}, title="Temperature Distribution")
        safe_plot(fig_temp_hist, key="ai_temp_hist")
        corr = df_out[["voltage","current","temperature","soc","battery_health_score"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Parameter Correlation Heatmap")
        safe_plot(fig_corr, key="ai_corr_heatmap")

    with tab_trad:
        st.subheader("Traditional BMS Insights")
        fig_v = px.line(df_out, x="time", y="voltage", labels={"time":"Time","voltage":"Voltage (V)"}, title="Voltage Over Time")
        safe_plot(fig_v, key="trad_voltage_chart")
        fig_t = px.line(df_out, x="time", y="temperature", labels={"time":"Time","temperature":"Temperature (°C)"}, title="Temperature Over Time")
        safe_plot(fig_t, key="trad_temp_chart")
        fig_soc_trad = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"SOC (%)"}, title="SOC Over Time (Traditional BMS)")
        safe_plot(fig_soc_trad, key="trad_soc_chart")

    with tab_compare:
        st.header("Compare — Combined result (CellGuard.AI first, then Traditional BMS)")
        st.markdown("### CellGuard.AI (Predictive view)")
        st.write(f"- Health Score: **{avg_score:.1f}/100**")
        st.write(f"- AI Anomaly %: **{anomaly_pct:.1f}%**")
        if alerts:
            st.write("- Current AI warnings:")
            for a in alerts:
                st.write(f"  - **{a['title']}** — {a['detail']}")
        else:
            st.write("- No AI warnings detected.")
        st.markdown("### Traditional BMS (Instant/raw view)")
        trad_cols = st.columns(3)
        with trad_cols[0]:
            if "voltage" in df_out.columns and df_out["voltage"].notna().sum()>0:
                st.metric("Voltage (mean)", f"{df_out['voltage'].mean():.3f} V")
            else:
                st.write("Voltage: N/A")
        with trad_cols[1]:
            if "temperature" in df_out.columns and df_out["temperature"].notna().sum()>0:
                st.metric("Temperature (mean)", f"{df_out['temperature'].mean():.2f} °C")
            else:
                st.write("Temperature: N/A")
        with trad_cols[2]:
            if "soc" in df_out.columns and df_out['soc'].notna().sum()>0:
                st.metric("SOC (last)", f"{df_out['soc'].iloc[-1]:.1f}%")
            else:
                st.write("SOC: N/A")
        st.markdown("---")
        st.subheader("Combined Recommendation")
        high_alerts = [a for a in alerts if a.get("severity")=="high"]
        if high_alerts or avg_score < 60:
            st.error("Combined verdict: Immediate action required. Reduce load, avoid fast charging, and schedule inspection.")
        elif avg_score < 75:
            st.warning("Combined verdict: Monitor closely. Apply conservative charge/discharge limits.")
        else:
            st.success("Combined verdict: Pack is healthy. Continue normal operation but monitor trends.")

    with tab_table:
        st.header("Processed Data & Export")
        st.download_button("⬇️ Download full report CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_FullReport.csv", "text/csv", key="download_full_report")
        st.dataframe(df_out.head(500), use_container_width=True)

    st.caption("CellGuard.AI — demo scenarios: Generic, EV, Drone, Phone. Toggle scenarios in the sidebar to simulate field conditions for testing.")

if __name__ == "__main__":
    main()
