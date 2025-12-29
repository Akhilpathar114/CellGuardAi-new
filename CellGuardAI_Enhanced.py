#CellGuardAI_Enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import re
from datetime import datetime

st.set_page_config(page_title="CellGuard.AI - Enhanced Dashboard", layout="wide")

# ----------------------
# Data Loading & Preprocessing
# ----------------------
def load_and_preprocess_bms_data(df_raw):
    """Load and preprocess Shell BMS CSV data"""
    df = df_raw.copy()
    
    # Remove empty rows
    df = df.dropna(how='all')
    
    # Create proper timestamp from Date and Time
    if 'Date' in df.columns and 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['time'] = range(len(df))
    else:
        df['time'] = range(len(df))
    
    # Clean SOC column (remove % sign)
    if 'Soc' in df.columns:
        df['Soc'] = df['Soc'].astype(str).str.replace('%', '').astype(float, errors='ignore')
    
    # Rename columns to standard format
    column_mapping = {
        'Pack Vol': 'pack_voltage',
        'Curent': 'current',
        'Soc': 'soc',
        'Rem. Ah': 'remaining_capacity',
        'Full Cap': 'full_capacity',
        'Cycle': 'cycle',
        'Temp1': 'temp1',
        'Temp2': 'temp2',
        'Temp3': 'temp3',
        'Temp4': 'temp4',
        'C_Low': 'cell_low',
        'C_Mid': 'cell_mid',
        'C_High': 'cell_high',
        'C_N_Low': 'cell_num_low',
        'C_N_High': 'cell_num_high',
        'C_Diff': 'cell_diff',
        'Err_Stat': 'error_status',
        'CMS': 'cms_status',
        'DMS': 'dms_status',
        'Bat.No': 'battery_number'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Calculate average temperature
    temp_cols = [c for c in ['temp1', 'temp2', 'temp3', 'temp4'] if c in df.columns]
    if temp_cols:
        df['temperature'] = df[temp_cols].mean(axis=1)
    else:
        df['temperature'] = np.nan
    
    # Calculate average cell voltage from individual cells
    cell_cols = [f'Cell{i}' for i in range(1, 25) if f'Cell{i}' in df.columns]
    if cell_cols:
        df['avg_cell_voltage'] = df[cell_cols].mean(axis=1)
        df['min_cell_voltage'] = df[cell_cols].min(axis=1)
        df['max_cell_voltage'] = df[cell_cols].max(axis=1)
        df['cell_voltage_std'] = df[cell_cols].std(axis=1)
    
    # Use pack voltage as main voltage if avg_cell_voltage not available
    if 'pack_voltage' in df.columns:
        if 'avg_cell_voltage' not in df.columns:
            df['voltage'] = df['pack_voltage']
        else:
            df['voltage'] = df['avg_cell_voltage']
    elif 'avg_cell_voltage' in df.columns:
        df['voltage'] = df['avg_cell_voltage']
    else:
        df['voltage'] = np.nan
    
    # Calculate capacity degradation
    if 'remaining_capacity' in df.columns and 'full_capacity' in df.columns:
        df['capacity_ratio'] = df['remaining_capacity'] / df['full_capacity']
        df['capacity_degradation'] = (1 - df['capacity_ratio']) * 100
    
    return df


def normalize_cols(df):
    """Try to map different column namings to our standard names"""
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
    """Add missing columns with NaN"""
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ----------------------
# Enhanced Feature Engineering
# ----------------------
def make_enhanced_features(df, window=10):
    """Create advanced features for ML models"""
    df = df.copy()
    
    # Basic features
    df = ensure_cols_exist(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    
    # Voltage features
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_std"] = df["voltage"].rolling(window, min_periods=1).std().fillna(0)
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
        df["voltage_min_max_range"] = df["voltage"].rolling(window, min_periods=1).max() - df["voltage"].rolling(window, min_periods=1).min()
        df["voltage_acceleration"] = df["voltage_roc"].diff().fillna(0)
    else:
        for col in ["voltage_ma", "voltage_std", "voltage_roc", "voltage_var", "voltage_min_max_range", "voltage_acceleration"]:
            df[col] = np.nan
    
    # Temperature features
    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_std"] = df["temperature"].rolling(window, min_periods=1).std().fillna(0)
        df["temp_roc"] = df["temperature"].diff().fillna(0)
        df["temp_max"] = df["temperature"].rolling(window*2, min_periods=1).max()
        df["temp_min"] = df["temperature"].rolling(window*2, min_periods=1).min()
    else:
        for col in ["temp_ma", "temp_std", "temp_roc", "temp_max", "temp_min"]:
            df[col] = np.nan
    
    # Current features
    if df["current"].notna().sum() > 0:
        df["current_ma"] = df["current"].rolling(window, min_periods=1).mean()
        df["current_std"] = df["current"].rolling(window, min_periods=1).std().fillna(0)
        df["current_roc"] = df["current"].diff().fillna(0)
        df["current_abs"] = df["current"].abs()
    else:
        for col in ["current_ma", "current_std", "current_roc", "current_abs"]:
            df[col] = np.nan
    
    # SOC features
    if df["soc"].notna().sum() > 0:
        df["soc_ma"] = df["soc"].rolling(window, min_periods=1).mean()
        df["soc_std"] = df["soc"].rolling(window, min_periods=1).std().fillna(0)
        df["soc_roc"] = df["soc"].diff().fillna(0)
    else:
        for col in ["soc_ma", "soc_std", "soc_roc"]:
            df[col] = np.nan
    
    # Cell imbalance features (if available)
    if 'cell_diff' in df.columns:
        df["cell_imbalance"] = df["cell_diff"]
        df["cell_imbalance_ma"] = df["cell_diff"].rolling(window, min_periods=1).mean()
    elif 'cell_voltage_std' in df.columns:
        df["cell_imbalance"] = df["cell_voltage_std"]
        df["cell_imbalance_ma"] = df["cell_voltage_std"].rolling(window, min_periods=1).mean()
    else:
        df["cell_imbalance"] = np.nan
        df["cell_imbalance_ma"] = np.nan
    
    # Capacity features (if available)
    if 'capacity_degradation' in df.columns:
        df["capacity_deg_ma"] = df["capacity_degradation"].rolling(window*3, min_periods=1).mean()
        df["capacity_deg_roc"] = df["capacity_degradation"].diff().fillna(0)
    else:
        df["capacity_deg_ma"] = np.nan
        df["capacity_deg_roc"] = np.nan
    
    # Risk labeling with enhanced logic
    create_risk_labels(df)
    
    return df


def create_risk_labels(df):
    """Create risk labels based on multiple conditions"""
    risk_conditions = pd.Series(False, index=df.index)
    
    # Voltage anomalies
    if df["voltage"].notna().sum() > 0:
        v_mean = df["voltage"].mean()
        v_std = df["voltage"].std()
        if not np.isnan(v_mean) and not np.isnan(v_std):
            risk_conditions |= (df["voltage"] < v_mean - 2*v_std)
            risk_conditions |= (df["voltage"] > v_mean + 2*v_std)
        if "voltage_roc" in df.columns:
            risk_conditions |= (df["voltage_roc"] < -0.05)
    
    # Temperature anomalies
    if df["temperature"].notna().sum() > 0:
        t_mean = df["temperature"].mean()
        t_std = df["temperature"].std()
        if not np.isnan(t_mean) and not np.isnan(t_std):
            risk_conditions |= (df["temperature"] > t_mean + 2.5*t_std)
        if "temp_roc" in df.columns:
            risk_conditions |= (df["temp_roc"] > 2.0)
    
    # SOC anomalies
    if "soc_roc" in df.columns:
        risk_conditions |= (df["soc_roc"] < -8)
        risk_conditions |= (df["soc_roc"] > 8)
    
    # Cell imbalance
    if "cell_imbalance" in df.columns and df["cell_imbalance"].notna().sum() > 0:
        ci_mean = df["cell_imbalance"].mean()
        ci_std = df["cell_imbalance"].std()
        if not np.isnan(ci_mean) and not np.isnan(ci_std):
            risk_conditions |= (df["cell_imbalance"] > ci_mean + 2*ci_std)
    
    # Error status
    if "error_status" in df.columns:
        risk_conditions |= (df["error_status"] > 0)
    
    df["risk_label"] = risk_conditions.astype(int)


# ----------------------
# Enhanced ML Models
# ----------------------
def run_enhanced_models(df, contamination=0.05):
    """Run multiple ML models for comprehensive analysis"""
    df = df.copy()
    
    # Select features for modeling
    feature_cols = [
        "voltage", "current", "temperature", "soc",
        "voltage_ma", "voltage_roc", "voltage_var",
        "temp_ma", "temp_roc", "temp_std",
        "current_ma", "current_std",
        "soc_ma", "soc_roc",
        "cell_imbalance"
    ]
    
    # Filter available features
    available_features = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 10]
    
    if len(available_features) < 3:
        st.warning("Insufficient features for ML models. Using basic analysis.")
        df["anomaly_flag"] = 0
        df["risk_pred"] = 0
        df["battery_health_score"] = 75.0
        df["failure_probability"] = 0.1
        df["degradation_rate"] = 0.0
        return df
    
    X = df[available_features].fillna(df[available_features].mean())
    
    # 1. Isolation Forest for Anomaly Detection
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        df["anomaly_flag"] = (iso_forest.fit_predict(X) == -1).astype(int)
        df["anomaly_score"] = -iso_forest.score_samples(X)
    except Exception as e:
        st.warning(f"Isolation Forest failed: {e}")
        df["anomaly_flag"] = 0
        df["anomaly_score"] = 0.0
    
    # 2. Risk Prediction using ensemble
    if "risk_label" in df.columns and df["risk_label"].sum() > 5:
        try:
            # Split train/test
            train_idx = df["risk_label"].notna()
            X_train = X[train_idx]
            y_train = df.loc[train_idx, "risk_label"]
            
            # Gradient Boosting Classifier
            gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4)
            gb_clf.fit(X_train, y_train)
            df["risk_pred"] = gb_clf.predict(X)
            df["risk_probability"] = gb_clf.predict_proba(X)[:, 1]
            
            # Random Forest for feature importance
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            rf_clf.fit(X_train, y_train)
            df["rf_risk_pred"] = rf_clf.predict(X)
            
            # Store feature importance for later analysis
            if not hasattr(st.session_state, 'feature_importance'):
                st.session_state.feature_importance = pd.DataFrame({
                    'feature': available_features,
                    'importance': rf_clf.feature_importances_
                }).sort_values('importance', ascending=False)
            
        except Exception as e:
            st.warning(f"Risk prediction model failed: {e}")
            df["risk_pred"] = 0
            df["risk_probability"] = 0.0
            df["rf_risk_pred"] = 0
    else:
        df["risk_pred"] = 0
        df["risk_probability"] = 0.0
        df["rf_risk_pred"] = 0
    
    # 3. Battery Health Score (composite metric)
    calculate_health_score(df)
    
    # 4. Failure Probability Estimation
    calculate_failure_probability(df)
    
    # 5. Degradation Rate Analysis
    calculate_degradation_rate(df)
    
    # 6. Clustering for pattern detection
    if len(df) > 50:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=min(4, len(df)//20), random_state=42)
            df["cluster"] = kmeans.fit_predict(X_scaled)
        except:
            df["cluster"] = 0
    else:
        df["cluster"] = 0
    
    return df


def calculate_health_score(df):
    """Calculate comprehensive battery health score (0-100)"""
    scores = []
    
    # Voltage health (30% weight)
    if df["voltage"].notna().sum() > 0:
        v_health = 100 - (df["voltage_var"] / df["voltage_var"].max() * 100) if df["voltage_var"].max() > 0 else 90
        v_health = v_health.fillna(90).clip(0, 100)
        scores.append(v_health * 0.30)
    
    # Temperature health (25% weight)
    if df["temperature"].notna().sum() > 0:
        safe_temp_range = (20, 40)
        t_dev = np.abs(df["temperature"] - np.mean(safe_temp_range))
        t_health = 100 - (t_dev / 30 * 100)
        t_health = t_health.clip(0, 100)
        scores.append(t_health * 0.25)
    
    # SOC health (20% weight)
    if df["soc"].notna().sum() > 0:
        soc_health = 100 - (df["soc_std"] if "soc_std" in df.columns else 0) * 2
        soc_health = pd.Series(soc_health).fillna(85).clip(0, 100)
        scores.append(soc_health * 0.20)
    
    # Cell balance health (15% weight)
    if "cell_imbalance" in df.columns and df["cell_imbalance"].notna().sum() > 0:
        max_imbalance = df["cell_imbalance"].max()
        if max_imbalance > 0:
            balance_health = 100 - (df["cell_imbalance"] / max_imbalance * 100)
        else:
            balance_health = 95
        balance_health = pd.Series(balance_health).fillna(95).clip(0, 100)
        scores.append(balance_health * 0.15)
    
    # Anomaly penalty (10% weight)
    anomaly_penalty = (1 - df["anomaly_flag"]) * 100 if "anomaly_flag" in df.columns else 90
    scores.append(pd.Series(anomaly_penalty) * 0.10)
    
    # Combine scores
    if scores:
        df["battery_health_score"] = sum(scores)
    else:
        df["battery_health_score"] = 75.0
    
    df["battery_health_score"] = df["battery_health_score"].clip(0, 100)


def calculate_failure_probability(df):
    """Estimate probability of failure in near future"""
    risk_factors = 0.0
    
    if "risk_pred" in df.columns:
        risk_factors += df["risk_pred"] * 0.3
    
    if "anomaly_flag" in df.columns:
        risk_factors += df["anomaly_flag"] * 0.2
    
    if "voltage_roc" in df.columns:
        risk_factors += (df["voltage_roc"] < -0.05).astype(float) * 0.2
    
    if "temp_roc" in df.columns:
        risk_factors += (df["temp_roc"] > 2.0).astype(float) * 0.15
    
    if "cell_imbalance" in df.columns and df["cell_imbalance"].notna().sum() > 0:
        ci_norm = df["cell_imbalance"] / (df["cell_imbalance"].max() + 1e-6)
        risk_factors += ci_norm * 0.15
    
    df["failure_probability"] = risk_factors.clip(0, 1)


def calculate_degradation_rate(df):
    """Calculate battery degradation rate over time"""
    if "capacity_degradation" in df.columns and df["capacity_degradation"].notna().sum() > 10:
        # Linear regression on capacity degradation
        valid_idx = df["capacity_degradation"].notna()
        if valid_idx.sum() > 10:
            X_time = df.loc[valid_idx, "time"].values.reshape(-1, 1)
            y_deg = df.loc[valid_idx, "capacity_degradation"].values
            
            try:
                lr = LinearRegression()
                lr.fit(X_time, y_deg)
                df["degradation_rate"] = lr.coef_[0]
            except:
                df["degradation_rate"] = 0.0
        else:
            df["degradation_rate"] = 0.0
    elif "voltage" in df.columns and df["voltage"].notna().sum() > 20:
        # Estimate from voltage trend
        valid_idx = df["voltage"].notna()
        if valid_idx.sum() > 20:
            X_time = df.loc[valid_idx, "time"].values.reshape(-1, 1)
            y_volt = df.loc[valid_idx, "voltage"].values
            
            try:
                lr = LinearRegression()
                lr.fit(X_time, y_volt)
                # Negative slope indicates degradation
                df["degradation_rate"] = -lr.coef_[0] * 1000  # Scale for visibility
            except:
                df["degradation_rate"] = 0.0
        else:
            df["degradation_rate"] = 0.0
    else:
        df["degradation_rate"] = 0.0


# ----------------------
# Problem Diagnosis & Solutions
# ----------------------
def diagnose_problems(df):
    """Identify specific problems and provide solutions"""
    problems = []
    
    # Problem 1: Cell Imbalance
    if "cell_imbalance" in df.columns and df["cell_imbalance"].notna().sum() > 0:
        avg_imbalance = df["cell_imbalance"].mean()
        max_imbalance = df["cell_imbalance"].max()
        
        if max_imbalance > 0.01:  # Threshold for concern
            severity = "high" if max_imbalance > 0.02 else "medium"
            problems.append({
                "problem": "Cell Voltage Imbalance Detected",
                "severity": severity,
                "details": f"Average imbalance: {avg_imbalance:.4f}V, Max: {max_imbalance:.4f}V",
                "impact": "Reduced capacity, uneven aging, potential safety risk",
                "solution": [
                    "Enable cell balancing in BMS",
                    "Perform manual balancing charge (slow charge to 100%)",
                    "Check individual cell health",
                    "Consider cell replacement if imbalance persists"
                ],
                "urgency": "High" if severity == "high" else "Medium"
            })
    
    # Problem 2: Temperature Issues
    if df["temperature"].notna().sum() > 0:
        avg_temp = df["temperature"].mean()
        max_temp = df["temperature"].max()
        temp_variance = df["temperature"].std()
        
        if max_temp > 45:
            problems.append({
                "problem": "High Operating Temperature",
                "severity": "high",
                "details": f"Max temperature: {max_temp:.1f}°C (Safe range: 20-40°C)",
                "impact": "Accelerated degradation, safety hazard, reduced lifespan",
                "solution": [
                    "Improve cooling system",
                    "Reduce charge/discharge rates",
                    "Check for thermal runaway risk",
                    "Inspect thermal management system",
                    "Ensure adequate ventilation"
                ],
                "urgency": "Critical"
            })
        elif temp_variance > 5:
            problems.append({
                "problem": "High Temperature Fluctuation",
                "severity": "medium",
                "details": f"Temperature std dev: {temp_variance:.2f}°C",
                "impact": "Thermal stress, uneven cell aging",
                "solution": [
                    "Check thermal sensor calibration",
                    "Improve temperature distribution",
                    "Review cooling system performance"
                ],
                "urgency": "Medium"
            })
    
    # Problem 3: Voltage Degradation
    if "voltage_roc" in df.columns:
        severe_drops = (df["voltage_roc"] < -0.1).sum()
        if severe_drops > len(df) * 0.05:  # More than 5% of readings
            problems.append({
                "problem": "Frequent Voltage Drops",
                "severity": "high",
                "details": f"Detected {severe_drops} severe voltage drops",
                "impact": "Possible internal resistance increase, cell degradation",
                "solution": [
                    "Measure internal resistance (ESR)",
                    "Reduce load current",
                    "Check for loose connections",
                    "Consider capacity test",
                    "Inspect for physical damage"
                ],
                "urgency": "High"
            })
    
    # Problem 4: Capacity Degradation
    if "capacity_degradation" in df.columns and df["capacity_degradation"].notna().sum() > 0:
        avg_deg = df["capacity_degradation"].mean()
        if avg_deg > 20:
            problems.append({
                "problem": "Significant Capacity Loss",
                "severity": "high",
                "details": f"Capacity degradation: {avg_deg:.1f}%",
                "impact": "Reduced runtime, end of useful life approaching",
                "solution": [
                    "Perform full capacity test",
                    "Review charge/discharge history",
                    "Consider battery replacement",
                    "Adjust usage patterns to preserve remaining capacity",
                    "Implement SOC limits (20-80%)"
                ],
                "urgency": "High"
            })
        elif avg_deg > 10:
            problems.append({
                "problem": "Moderate Capacity Degradation",
                "severity": "medium",
                "details": f"Capacity degradation: {avg_deg:.1f}%",
                "impact": "Normal aging, monitor for acceleration",
                "solution": [
                    "Continue regular monitoring",
                    "Optimize charge/discharge patterns",
                    "Avoid extreme SOC levels",
                    "Maintain optimal temperature range"
                ],
                "urgency": "Medium"
            })
    
    # Problem 5: High Anomaly Rate
    if "anomaly_flag" in df.columns:
        anomaly_rate = df["anomaly_flag"].mean() * 100
        if anomaly_rate > 10:
            problems.append({
                "problem": "High Anomaly Detection Rate",
                "severity": "high",
                "details": f"Anomaly rate: {anomaly_rate:.1f}%",
                "impact": "Unpredictable behavior, potential system instability",
                "solution": [
                    "Investigate root cause of anomalies",
                    "Review BMS sensor calibration",
                    "Check for intermittent faults",
                    "Perform comprehensive system diagnosis",
                    "Consider BMS firmware update"
                ],
                "urgency": "High"
            })
    
    # Problem 6: SOC Instability
    if "soc_roc" in df.columns:
        extreme_soc_changes = (df["soc_roc"].abs() > 5).sum()
        if extreme_soc_changes > len(df) * 0.1:
            problems.append({
                "problem": "SOC Estimation Instability",
                "severity": "medium",
                "details": f"Detected {extreme_soc_changes} extreme SOC changes",
                "impact": "Inaccurate range estimation, possible coulomb counting error",
                "solution": [
                    "Recalibrate SOC algorithm",
                    "Perform full charge-discharge cycle",
                    "Update BMS parameters",
                    "Check current sensor accuracy"
                ],
                "urgency": "Medium"
            })
    
    # Problem 7: Cycle Count Analysis
    if "cycle" in df.columns and df["cycle"].notna().sum() > 0:
        max_cycle = df["cycle"].max()
        if max_cycle > 2000:
            problems.append({
                "problem": "High Cycle Count",
                "severity": "medium",
                "details": f"Cycle count: {int(max_cycle)}",
                "impact": "Expected capacity fade, nearing rated life",
                "solution": [
                    "Plan for eventual replacement",
                    "Reduce DOD (depth of discharge)",
                    "Perform periodic health checks",
                    "Monitor for accelerated degradation"
                ],
                "urgency": "Low"
            })
    
    # Sort by urgency
    urgency_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    problems.sort(key=lambda x: urgency_order.get(x["urgency"], 4))
    
    return problems


def generate_recommendations(df, problems):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    # Based on health score
    avg_health = df["battery_health_score"].mean()
    
    if avg_health < 60:
        recommendations.append({
            "category": "Critical Action Required",
            "recommendation": "Immediate battery inspection and potential replacement",
            "rationale": f"Overall health score ({avg_health:.1f}) indicates severe degradation"
        })
    elif avg_health < 75:
        recommendations.append({
            "category": "Preventive Maintenance",
            "recommendation": "Schedule comprehensive battery check within 1 week",
            "rationale": f"Health score ({avg_health:.1f}) shows concerning trends"
        })
    
    # Based on failure probability
    if "failure_probability" in df.columns:
        max_fail_prob = df["failure_probability"].max()
        if max_fail_prob > 0.7:
            recommendations.append({
                "category": "Safety Alert",
                "recommendation": "Reduce load and implement safety protocols",
                "rationale": f"High failure probability detected ({max_fail_prob:.1%})"
            })
    
    # Based on specific problems
    for prob in problems[:3]:  # Top 3 problems
        recommendations.append({
            "category": prob["problem"],
            "recommendation": prob["solution"][0],  # Primary solution
            "rationale": prob["details"]
        })
    
    return recommendations


# ----------------------
# Visualization Helpers
# ----------------------
def make_gauge_figure(score):
    """Create gauge chart for health score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        delta={'reference': 85},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "#ffcccc"},
                {'range': [60, 75], 'color': "#fff4cc"},
                {'range': [75, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def safe_plot(fig, key):
    """Safely render plotly chart"""
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.error(f"Chart error: {e}")


def pack_label(score):
    """Determine pack status label"""
    if score >= 75:
        return "HEALTHY", "#2ecc71"
    elif score >= 60:
        return "WATCH", "#f39c12"
    else:
        return "CRITICAL", "#e74c3c"


# ----------------------
# Enhanced PDF Report
# ----------------------
def make_enhanced_pdf(df, avg_score, anomaly_pct, problems, recommendations, verdict):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "CellGuard.AI - Enhanced Battery Analysis Report")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary Section
    y = height - 110
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Executive Summary")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Health Score: {avg_score:.1f}/100")
    y -= 15
    c.drawString(50, y, f"Anomaly Rate: {anomaly_pct:.1f}%")
    y -= 15
    c.drawString(50, y, f"Verdict: {verdict}")
    y -= 30
    
    # Problems Section
    if problems:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Identified Problems")
        y -= 20
        
        for i, prob in enumerate(problems[:5], 1):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, f"{i}. {prob['problem']} [{prob['urgency']}]")
            y -= 15
            c.setFont("Helvetica", 9)
            c.drawString(70, y, f"Details: {prob['details']}")
            y -= 12
            c.drawString(70, y, f"Solution: {prob['solution'][0]}")
            y -= 20
            
            if y < 100:
                c.showPage()
                y = height - 50
    
    # Recommendations Section
    if recommendations:
        if y < 200:
            c.showPage()
            y = height - 50
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Recommendations")
        y -= 20
        
        for i, rec in enumerate(recommendations[:5], 1):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, f"{i}. {rec['category']}")
            y -= 15
            c.setFont("Helvetica", 9)
            c.drawString(70, y, rec['recommendation'])
            y -= 15
            
            if y < 100:
                c.showPage()
                y = height - 50
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------
# Main Application
# ----------------------
def main():
    st.title("🔋 CellGuard.AI - Enhanced Battery Management System")
    st.markdown("*Advanced ML-powered Battery Diagnostics & Predictive Maintenance*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Synthetic Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload BMS Data (CSV)", type=["csv"])
        
        if uploaded_file is None:
            st.info("Please upload a CSV file to begin analysis")
            st.markdown("### Expected CSV Format")
            st.markdown("""
            The system supports Shell BMS format with columns like:
            - Date, Time
            - Pack Vol, Current, SOC
            - Cell1-Cell24 (individual cell voltages)
            - Temp1-Temp4
            - Cycle, Error Status
            - C_Low, C_High, C_Diff (cell statistics)
            """)
            return
        
        df_raw = pd.read_csv(uploaded_file)
        df_raw = load_and_preprocess_bms_data(df_raw)
        
    else:
        scenario = st.sidebar.selectbox("Select Scenario", ["Generic", "EV", "Drone", "Phone"])
        n_points = st.sidebar.slider("Data Points", 200, 2000, 800)
        df_raw = gen_sample_data(n=n_points, scenario=scenario)
        df_raw, _ = normalize_cols(df_raw)
    
    # Processing parameters
    st.sidebar.markdown("### Processing Parameters")
    window = st.sidebar.slider("Rolling Window", 5, 30, 10)
    contamination = st.sidebar.slider("Anomaly Contamination", 0.01, 0.20, 0.05)
    
    if st.sidebar.button("🚀 Run Enhanced Analysis", type="primary"):
        with st.spinner("Running advanced ML analysis..."):
            # Feature engineering
            df_feat = make_enhanced_features(df_raw, window=window)
            
            # ML models
            df_out = run_enhanced_models(df_feat, contamination=contamination)
            
            # Problem diagnosis
            problems = diagnose_problems(df_out)
            recommendations = generate_recommendations(df_out, problems)
            
            # Store in session state
            st.session_state.df_result = df_out
            st.session_state.problems = problems
            st.session_state.recommendations = recommendations
            st.success("Analysis complete!")
    
    if 'df_result' not in st.session_state:
        st.info("Click 'Run Enhanced Analysis' to start")
        return
    
    df_out = st.session_state.df_result
    problems = st.session_state.problems
    recommendations = st.session_state.recommendations
    
    # Calculate metrics
    avg_score = float(df_out["battery_health_score"].mean())
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100)
    label, color = pack_label(avg_score)
    
    # Verdict
    if avg_score < 60:
        verdict = "Critical: Immediate action required"
    elif avg_score < 75:
        verdict = "Warning: Monitor closely"
    else:
        verdict = "Healthy: Normal operation"
    
    # Generate PDF
    pdf_bytes = make_enhanced_pdf(df_out, avg_score, anomaly_pct, problems, recommendations, verdict)
    
    # ===== DASHBOARD LAYOUT =====
    
    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🏥 Battery Health")
        gauge = make_gauge_figure(avg_score)
        safe_plot(gauge, key="main_gauge")
    
    with col2:
        st.markdown("### 📊 Status")
        st.markdown(f"<div style='background:{color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:20px;font-weight:bold'>{label}</div>", unsafe_allow_html=True)
        st.metric("Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f}")
    
    with col3:
        st.markdown("### ⚠️ Anomalies")
        st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")
        st.metric("Data Points", f"{len(df_out):,}")
    
    with col4:
        st.markdown("### 🔍 Analysis")
        st.metric("Problems Found", len(problems))
        st.metric("Critical Issues", sum(1 for p in problems if p['urgency'] in ['Critical', 'High']))
    
    # Downloads
    st.markdown("---")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="CellGuardAI_Enhanced_Report.pdf",
            mime="application/pdf"
        )
    with col_dl2:
        st.download_button(
            "📊 Download Analysis CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="CellGuardAI_Enhanced_Analysis.csv",
            mime="text/csv"
        )
    with col_dl3:
        if 'feature_importance' in st.session_state:
            st.download_button(
                "📈 Download Feature Importance",
                data=st.session_state.feature_importance.to_csv(index=False).encode("utf-8"),
                file_name="Feature_Importance.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Problem Diagnosis Section
    st.header("🔧 Problem Diagnosis & Solutions")
    
    if problems:
        for i, prob in enumerate(problems, 1):
            urgency_colors = {
                "Critical": "#e74c3c",
                "High": "#e67e22",
                "Medium": "#f39c12",
                "Low": "#3498db"
            }
            prob_color = urgency_colors.get(prob['urgency'], "#95a5a6")
            
            with st.expander(f"{'🔴' if prob['urgency']=='Critical' else '🟠' if prob['urgency']=='High' else '🟡' if prob['urgency']=='Medium' else '🔵'} Problem {i}: {prob['problem']} [{prob['urgency']}]", expanded=(i<=2)):
                col_p1, col_p2 = st.columns([1, 1])
                
                with col_p1:
                    st.markdown("**Details:**")
                    st.info(prob['details'])
                    st.markdown("**Impact:**")
                    st.warning(prob['impact'])
                
                with col_p2:
                    st.markdown("**Recommended Solutions:**")
                    for j, sol in enumerate(prob['solution'], 1):
                        st.markdown(f"{j}. {sol}")
    else:
        st.success("✅ No critical problems detected. System is operating normally.")
    
    st.markdown("---")
    
    # Recommendations Section
    st.header("💡 Actionable Recommendations")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {rec['category']}**")
            st.write(f"   • {rec['recommendation']}")
            st.caption(f"   Rationale: {rec['rationale']}")
    else:
        st.info("Continue regular monitoring and maintenance schedule")
    
    st.markdown("---")
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 AI Insights",
        "📊 Traditional Metrics",
        "🔬 Advanced Analytics",
        "📉 Trends & Predictions",
        "📋 Data Table"
    ])
    
    with tab1:
        st.subheader("AI-Powered Insights")
        
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            # Health score over time
            fig_health = px.line(
                df_out,
                x="time",
                y="battery_health_score",
                title="Battery Health Score Timeline",
                labels={"time": "Time Index", "battery_health_score": "Health Score"}
            )
            fig_health.add_hline(y=75, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
            fig_health.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
            safe_plot(fig_health, key="health_timeline")
            
            # Anomaly detection
            fig_anomaly = px.scatter(
                df_out,
                x="time",
                y="voltage",
                color="anomaly_flag",
                title="Anomaly Detection (Voltage)",
                labels={"time": "Time Index", "voltage": "Voltage (V)", "anomaly_flag": "Anomaly"},
                color_discrete_map={0: "blue", 1: "red"}
            )
            safe_plot(fig_anomaly, key="anomaly_scatter")
        
        with col_ai2:
            # Risk probability
            if "risk_probability" in df_out.columns:
                fig_risk = px.line(
                    df_out,
                    x="time",
                    y="risk_probability",
                    title="Risk Probability Over Time",
                    labels={"time": "Time Index", "risk_probability": "Risk Probability"}
                )
                fig_risk.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
                safe_plot(fig_risk, key="risk_prob")
            
            # Failure probability
            if "failure_probability" in df_out.columns:
                fig_fail = px.area(
                    df_out,
                    x="time",
                    y="failure_probability",
                    title="Failure Probability Trend",
                    labels={"time": "Time Index", "failure_probability": "Failure Probability"}
                )
                safe_plot(fig_fail, key="failure_prob")
        
        # Correlation heatmap
        st.subheader("Parameter Correlations")
        corr_cols = ["voltage", "current", "temperature", "soc", "battery_health_score"]
        available_corr = [c for c in corr_cols if c in df_out.columns]
        if len(available_corr) >= 3:
            corr_matrix = df_out[available_corr].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r"
            )
            safe_plot(fig_corr, key="correlation_heatmap")
    
    with tab2:
        st.subheader("Traditional BMS Metrics")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            # Voltage
            if "voltage" in df_out.columns:
                fig_volt = px.line(
                    df_out,
                    x="time",
                    y="voltage",
                    title="Voltage Over Time",
                    labels={"time": "Time Index", "voltage": "Voltage (V)"}
                )
                safe_plot(fig_volt, key="voltage_line")
            
            # SOC
            if "soc" in df_out.columns:
                fig_soc = px.line(
                    df_out,
                    x="time",
                    y="soc",
                    title="State of Charge (SOC)",
                    labels={"time": "Time Index", "soc": "SOC (%)"}
                )
                safe_plot(fig_soc, key="soc_line")
        
        with col_t2:
            # Temperature
            if "temperature" in df_out.columns:
                fig_temp = px.line(
                    df_out,
                    x="time",
                    y="temperature",
                    title="Temperature Over Time",
                    labels={"time": "Time Index", "temperature": "Temperature (°C)"}
                )
                fig_temp.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Upper Safe Limit")
                safe_plot(fig_temp, key="temp_line")
            
            # Current
            if "current" in df_out.columns:
                fig_curr = px.line(
                    df_out,
                    x="time",
                    y="current",
                    title="Current Flow",
                    labels={"time": "Time Index", "current": "Current (A)"}
                )
                safe_plot(fig_curr, key="current_line")
    
    with tab3:
        st.subheader("Advanced Analytics")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            # Cell imbalance
            if "cell_imbalance" in df_out.columns and df_out["cell_imbalance"].notna().sum() > 0:
                fig_imb = px.line(
                    df_out,
                    x="time",
                    y="cell_imbalance",
                    title="Cell Voltage Imbalance",
                    labels={"time": "Time Index", "cell_imbalance": "Imbalance (V)"}
                )
                safe_plot(fig_imb, key="cell_imbalance")
            
            # Voltage variance
            if "voltage_var" in df_out.columns:
                fig_var = px.line(
                    df_out,
                    x="time",
                    y="voltage_var",
                    title="Voltage Variance",
                    labels={"time": "Time Index", "voltage_var": "Variance"}
                )
                safe_plot(fig_var, key="voltage_var")
        
        with col_a2:
            # Clustering visualization
            if "cluster" in df_out.columns and len(df_out) > 50:
                if "voltage" in df_out.columns and "temperature" in df_out.columns:
                    fig_cluster = px.scatter(
                        df_out,
                        x="voltage",
                        y="temperature",
                        color="cluster",
                        title="Operational Pattern Clusters",
                        labels={"voltage": "Voltage (V)", "temperature": "Temperature (°C)"}
                    )
                    safe_plot(fig_cluster, key="clustering")
            
            # Temperature distribution
            if "temperature" in df_out.columns:
                fig_temp_hist = px.histogram(
                    df_out,
                    x="temperature",
                    title="Temperature Distribution",
                    labels={"temperature": "Temperature (°C)"}
                )
                safe_plot(fig_temp_hist, key="temp_histogram")
        
        # Feature importance
        if 'feature_importance' in st.session_state:
            st.subheader("Feature Importance for Risk Prediction")
            fig_imp = px.bar(
                st.session_state.feature_importance.head(10),
                x="importance",
                y="feature",
                orientation='h',
                title="Top 10 Most Important Features"
            )
            safe_plot(fig_imp, key="feature_importance")
    
    with tab4:
        st.subheader("Trends & Predictive Analytics")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            # Degradation rate
            if "degradation_rate" in df_out.columns:
                avg_deg_rate = df_out["degradation_rate"].mean()
                st.metric("Average Degradation Rate", f"{avg_deg_rate:.4f}% per cycle")
                
                if "capacity_degradation" in df_out.columns:
                    fig_deg = px.line(
                        df_out,
                        x="time",
                        y="capacity_degradation",
                        title="Capacity Degradation Trend",
                        labels={"time": "Time Index", "capacity_degradation": "Degradation (%)"}
                    )
                    safe_plot(fig_deg, key="capacity_degradation")
        
        with col_p2:
            # Voltage rate of change
            if "voltage_roc" in df_out.columns:
                fig_roc = px.line(
                    df_out,
                    x="time",
                    y="voltage_roc",
                    title="Voltage Rate of Change",
                    labels={"time": "Time Index", "voltage_roc": "dV/dt"}
                )
                safe_plot(fig_roc, key="voltage_roc")
        
        # Summary metrics
        st.subheader("Statistical Summary")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            if "voltage" in df_out.columns:
                st.metric("Avg Voltage", f"{df_out['voltage'].mean():.3f} V")
        with metrics_col2:
            if "temperature" in df_out.columns:
                st.metric("Avg Temperature", f"{df_out['temperature'].mean():.2f} °C")
        with metrics_col3:
            if "soc" in df_out.columns:
                st.metric("Current SOC", f"{df_out['soc'].iloc[-1]:.1f}%")
        with metrics_col4:
            if "cycle" in df_out.columns:
                st.metric("Cycle Count", f"{int(df_out['cycle'].max())}")
    
    with tab5:
        st.subheader("Processed Data Export")
        
        # Show sample data
        st.dataframe(df_out.head(100), use_container_width=True)
        
        st.caption(f"Showing first 100 rows of {len(df_out)} total records")
    
    # Footer
    st.markdown("---")
    st.caption("CellGuard.AI Enhanced - Advanced ML-powered Battery Diagnostics | Predictive Maintenance System")


# Simple synthetic data generator (for demo)
def gen_sample_data(n=800, seed=42, scenario="Generic"):
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


if __name__ == "__main__":
    main()
