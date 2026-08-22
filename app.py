"""
Kenya Port Throughput Forecast — Streamlit App
------------------------------------------------
Two ways to get a forecast:
  1. Manual inputs  -- you type in vessel calls & cargo volumes yourself
  2. Pick a date     -- the app fills in typical seasonal averages for you

This app does NOT train any models. It only loads the already-trained
pipeline objects (produced by the notebook) and runs them on new inputs.

Required files, in the same folder as this script:
  preprocessor.pkl
  stage1_classifier.pkl
  stage2_rf.pkl
  stage2_lgb.pkl
  stage2_svr.pkl
  stage2_scaler.pkl
  meta_ridge.pkl
  seasonal_lookup.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kenya Port Throughput Forecast", page_icon="⚓", layout="wide")

RAW_INPUT_COLS = [
    'portcalls_container', 'portcalls_dry_bulk', 'portcalls_general_cargo',
    'portcalls_roro', 'portcalls_tanker',
    'import_container', 'import_dry_bulk', 'import_general_cargo',
    'import_roro', 'import_tanker',
    'export_container', 'export_dry_bulk', 'export_general_cargo',
    'export_roro', 'export_tanker',
]

FEATURES = [
    'portname',
    'portcalls_container', 'portcalls_dry_bulk', 'portcalls_general_cargo',
    'portcalls_roro', 'portcalls_tanker', 'total_portcalls',
    'import_container', 'import_dry_bulk', 'import_general_cargo',
    'import_roro', 'import_tanker',
    'export_container', 'export_dry_bulk', 'export_general_cargo',
    'export_roro', 'export_tanker',
    'tanker_call_ratio', 'container_call_ratio', 'bulk_call_ratio',
    'import_export_ratio',
    'month', 'quarter', 'day_of_week', 'is_weekend',
]


# ──────────────────────────────────────────────────────────────────────────
# Load model artifacts (cached so this only runs once per session)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    here = os.path.dirname(os.path.abspath(__file__))

    def p(name):
        return os.path.join(here, name)

    artifacts = {
        "preprocessor": joblib.load(p("preprocessor.pkl")),
        "clf":          joblib.load(p("stage1_classifier.pkl")),
        "rf":           joblib.load(p("stage2_rf.pkl")),
        "lgb_reg":      joblib.load(p("stage2_lgb.pkl")),
        "svr":          joblib.load(p("stage2_svr.pkl")),
        "scaler":       joblib.load(p("stage2_scaler.pkl")),
        "ridge":        joblib.load(p("meta_ridge.pkl")),
    }
    seasonal_lookup = pd.read_csv(p("seasonal_lookup.csv")).set_index(["portname", "month"])
    return artifacts, seasonal_lookup


try:
    artifacts, seasonal_lookup = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not find one or more model files in this app's folder.\n\n"
        f"Missing: {e.filename}\n\n"
        "Make sure preprocessor.pkl, stage1_classifier.pkl, stage2_rf.pkl, "
        "stage2_lgb.pkl, stage2_svr.pkl, stage2_scaler.pkl, meta_ridge.pkl "
        "and seasonal_lookup.csv are all saved next to app.py."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# Feature engineering — must mirror the notebook's build_features() exactly
# ──────────────────────────────────────────────────────────────────────────
def build_features(df):
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])

    d['month'] = d['date'].dt.month
    d['quarter'] = d['date'].dt.quarter
    d['day_of_week'] = d['date'].dt.dayofweek
    d['is_weekend'] = d['day_of_week'].isin([5, 6]).astype(int)

    portcall_cols = ['portcalls_container', 'portcalls_dry_bulk',
                      'portcalls_general_cargo', 'portcalls_roro', 'portcalls_tanker']
    d['total_portcalls'] = d[portcall_cols].sum(axis=1)

    d['tanker_call_ratio'] = np.where(d['total_portcalls'] > 0,
                                       d['portcalls_tanker'] / d['total_portcalls'], 0)
    d['container_call_ratio'] = np.where(d['total_portcalls'] > 0,
                                          d['portcalls_container'] / d['total_portcalls'], 0)
    d['bulk_call_ratio'] = np.where(d['total_portcalls'] > 0,
                                     d['portcalls_dry_bulk'] / d['total_portcalls'], 0)

    d['import_export_ratio'] = np.where(
        (d['total_import'] + d['total_export']) > 0,
        d['total_import'] / (d['total_import'] + d['total_export']), 0.5)

    return d


# ──────────────────────────────────────────────────────────────────────────
# Core prediction pipeline — Stage 1 -> Stage 2A -> Stage 2B
# ──────────────────────────────────────────────────────────────────────────
def forecast_single_day(port, forecast_date, **inputs):
    preprocessor = artifacts["preprocessor"]
    clf = artifacts["clf"]
    rf = artifacts["rf"]
    lgb_reg = artifacts["lgb_reg"]
    svr = artifacts["svr"]
    scaler = artifacts["scaler"]
    ridge = artifacts["ridge"]

    raw = {col: inputs.get(col, 0) for col in RAW_INPUT_COLS}
    total_import = sum(raw[c] for c in RAW_INPUT_COLS if c.startswith("import_"))
    total_export = sum(raw[c] for c in RAW_INPUT_COLS if c.startswith("export_"))

    row = pd.DataFrame([{
        "date": pd.to_datetime(forecast_date),
        "portname": port,
        **raw,
        "total_import": total_import,
        "total_export": total_export,
    }])

    row = build_features(row)
    row_X = row[FEATURES]
    row_enc = preprocessor.transform(row_X)

    p_active = clf.predict_proba(row_enc)[:, 1][0]
    is_active = clf.predict(row_enc)[0]

    if is_active == 0:
        return {
            "prediction_tonnes": 0.0,
            "p_active": p_active,
            "stage": "Stage 1 predicted a zero-activity day",
            "inputs_used": raw,
        }

    row_scaled = scaler.transform(row_enc)
    p_rf = rf.predict(row_enc)[0]
    p_lgb = lgb_reg.predict(row_enc)[0]
    p_svr = svr.predict(row_scaled)[0]

    meta_row = np.array([[p_rf, p_lgb, p_svr, p_active]])
    final_pred = max(ridge.predict(meta_row)[0], 0)

    return {
        "prediction_tonnes": final_pred,
        "p_active": p_active,
        "base_predictions": {"Random Forest": p_rf, "LightGBM": p_lgb, "SVR": p_svr},
        "stage": "Stage 2 blend",
        "inputs_used": raw,
    }


def forecast_by_date(port, forecast_date):
    month = pd.to_datetime(forecast_date).month
    if (port, month) not in seasonal_lookup.index:
        raise ValueError(f"No historical seasonal data for {port}, month {month}")
    avg_inputs = seasonal_lookup.loc[(port, month)].to_dict()
    result = forecast_single_day(port, forecast_date, **avg_inputs)
    result["note"] = f"Based on {port}'s historical average conditions for month {month}"
    return result


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.title("⚓ Kenya Port Throughput Forecast")
st.caption("Two-stage zero-inflated stacking ensemble — Mombasa & Lamu")

mode = st.radio(
    "Forecast mode",
    ["Pick a date (seasonal averages)", "Manual inputs"],
    horizontal=True,
)

port = st.selectbox("Select port", ["Mombasa", "Lamu"])

if mode == "Pick a date (seasonal averages)":
    forecast_date = st.date_input("Forecast date")

    if st.button("Run Forecast", type="primary"):
        try:
            result = forecast_by_date(port, forecast_date)
        except ValueError as e:
            st.error(str(e))
        else:
            st.metric("Predicted throughput", f"{result['prediction_tonnes']:,.0f} tonnes")
            st.caption(result["note"])
            st.caption(f"Model confidence this is an active day: {result['p_active']:.1%}")
            with st.expander("Assumed conditions used for this forecast"):
                st.json(result["inputs_used"])
            st.info(
                "This forecast assumes typical historical conditions for this "
                "port and month — it does not know about specific ships or "
                "cargo scheduled for this exact date.",
                icon="ℹ️",
            )

else:
    forecast_date = st.date_input("Forecast date")

    st.subheader("Port & vessel calls")
    c1, c2, c3, c4, c5 = st.columns(5)
    portcalls_container = c1.number_input("Container calls", min_value=0, value=1)
    portcalls_tanker = c2.number_input("Tanker calls", min_value=0, value=1)
    portcalls_dry_bulk = c3.number_input("Dry bulk calls", min_value=0, value=0)
    portcalls_general_cargo = c4.number_input("General cargo calls", min_value=0, value=0)
    portcalls_roro = c5.number_input("Ro-Ro calls", min_value=0, value=0)

    st.subheader("Import volumes (tonnes)")
    c1, c2, c3, c4, c5 = st.columns(5)
    import_tanker = c1.number_input("Tanker imports", min_value=0, value=0)
    import_dry_bulk = c2.number_input("Dry bulk imports", min_value=0, value=0)
    import_container = c3.number_input("Container imports", min_value=0, value=0)
    import_general_cargo = c4.number_input("General cargo imports", min_value=0, value=0)
    import_roro = c5.number_input("Ro-Ro imports", min_value=0, value=0)

    st.subheader("Export volumes (tonnes)")
    c1, c2, c3, c4, c5 = st.columns(5)
    export_tanker = c1.number_input("Tanker exports", min_value=0, value=0)
    export_dry_bulk = c2.number_input("Dry bulk exports", min_value=0, value=0)
    export_container = c3.number_input("Container exports", min_value=0, value=0)
    export_general_cargo = c4.number_input("General cargo exports", min_value=0, value=0)
    export_roro = c5.number_input("Ro-Ro exports", min_value=0, value=0)

    if st.button("Run Forecast", type="primary"):
        result = forecast_single_day(
            port, forecast_date,
            portcalls_container=portcalls_container, portcalls_tanker=portcalls_tanker,
            portcalls_dry_bulk=portcalls_dry_bulk, portcalls_general_cargo=portcalls_general_cargo,
            portcalls_roro=portcalls_roro,
            import_tanker=import_tanker, import_dry_bulk=import_dry_bulk,
            import_container=import_container, import_general_cargo=import_general_cargo,
            import_roro=import_roro,
            export_tanker=export_tanker, export_dry_bulk=export_dry_bulk,
            export_container=export_container, export_general_cargo=export_general_cargo,
            export_roro=export_roro,
        )
        st.metric("Predicted throughput", f"{result['prediction_tonnes']:,.0f} tonnes")
        st.caption(f"Model confidence this is an active day: {result['p_active']:.1%}")
        if "base_predictions" in result:
            with st.expander("Individual base model predictions"):
                st.json(result["base_predictions"])
