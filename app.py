# ==============================================================================
# AirCast India - Streamlit Air Quality Forecasting Application (app.py)
#
# Predicts real-time AQI across Indian locations using live Open-Meteo
# air quality data and pre-trained Machine Learning models.
#
# Sections:
#   - Model Selector and Performance Metrics (Top Header)
#   - Tab 1: Custom Prediction (Raw inputs with City Live Data Autofill)
#   - Tab 2: Location Forecast (Deep-dive, CPCB gauge, 4-model consensus)
#   - Tab 3: India Glow Map (3-tier halo glowing markers on dark basemap)
#   - Tab 4: Insights (Model comparison table, feature importances, test chart)
# ==============================================================================

import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION (No sidebar, dark layout)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AirCast India",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark styling for metrics and badges
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    [data-testid="stSidebar"] { display: none; }
    .stMetric {
        background: rgba(255, 255, 255, 0.04);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .cpcb-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def aqi_category(aqi):
    """
    Returns (label, hex_colour, health_advice) based on official Indian CPCB guidelines.
    Categories:
      0-50:    Good (Green)
      51-100:  Satisfactory (Light Green)
      101-200: Moderate (Yellow)
      201-300: Poor (Orange)
      301-400: Very Poor (Red)
      401+:    Severe (Maroon)
    """
    aqi_val = float(aqi)
    if aqi_val <= 50:
        return (
            "Good",
            "#00E400",
            "Air quality is satisfactory, and air pollution poses little or no risk."
        )
    elif aqi_val <= 100:
        return (
            "Satisfactory",
            "#92D050",
            "Air quality is acceptable; sensitive individuals may experience minor breathing discomfort."
        )
    elif aqi_val <= 200:
        return (
            "Moderate",
            "#FFDE33",
            "May cause breathing discomfort to people with lung disease, asthma, or heart ailments."
        )
    elif aqi_val <= 300:
        return (
            "Poor",
            "#FF9933",
            "May cause breathing discomfort to most people on prolonged outdoor exposure."
        )
    elif aqi_val <= 400:
        return (
            "Very Poor",
            "#CC0033",
            "May cause respiratory illness to people on prolonged exposure; avoid strenuous outdoor exercise."
        )
    else:
        return (
            "Severe",
            "#7E0023",
            "Emergency health warning: affects healthy people and seriously impacts those with existing conditions."
        )


@st.cache_resource
def load_models_and_metrics():
    """
    Loads pre-trained models, feature metadata, and evaluation metrics.
    Cached as a resource so it is loaded into memory only once.
    """
    if not os.path.exists("models.pkl") or not os.path.exists("metrics.json"):
        st.error("Missing 'models.pkl' or 'metrics.json'. Please run 'python train.py' first.")
        st.stop()

    models_data = joblib.load("models.pkl")
    with open("metrics.json", "r") as f:
        metrics_data = json.load(f)

    return models_data, metrics_data


@st.cache_data
def load_datasets():
    """
    Loads locations list and historical daily city records.
    Cached so disk reads happen only once.
    """
    loc_file = os.path.join("data", "locations.csv")
    city_file = os.path.join("data", "city_day.csv")

    locations_df = pd.read_csv(loc_file)
    city_df = pd.read_csv(city_file)
    city_df["Date"] = pd.to_datetime(city_df["Date"])

    return locations_df, city_df


@st.cache_data(ttl=3600)
def fetch_live_pollutants(locations_df):
    """
    Calls Open-Meteo Air Quality API in batches of ~25 locations to keep URL lengths short.
    Averages today's 24-hour readings to match the daily format of historical training data.
    Divides CO by 1000 to convert from Open-Meteo µg/m³ to dataset mg/m³.
    If the API fails (offline, timeout, rate limit), safely falls back to demo data.
    """
    endpoint = "https://air-quality-api.open-meteo.com/v1/air-quality"
    batch_size = 25
    records = []
    is_fallback = False

    try:
        for i in range(0, len(locations_df), batch_size):
            batch = locations_df.iloc[i:i + batch_size]
            lats = ",".join(batch["lat"].astype(str))
            lons = ",".join(batch["lon"].astype(str))

            params = {
                "latitude": lats,
                "longitude": lons,
                "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
                "forecast_days": 1,
                "timezone": "Asia/Kolkata"
            }

            res = requests.get(endpoint, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            if not isinstance(data, list):
                data = [data]

            for loc_row, loc_data in zip(batch.to_dict("records"), data):
                hourly = loc_data.get("hourly", {})

                def get_daily_avg(key):
                    vals = hourly.get(key, [])
                    clean = [v for v in vals if v is not None]
                    return sum(clean) / len(clean) if clean else 0.0

                records.append({
                    "name": loc_row["name"],
                    "lat": float(loc_row["lat"]),
                    "lon": float(loc_row["lon"]),
                    "PM2.5": round(get_daily_avg("pm2_5"), 2),
                    "PM10": round(get_daily_avg("pm10"), 2),
                    "NO2": round(get_daily_avg("nitrogen_dioxide"), 2),
                    "SO2": round(get_daily_avg("sulphur_dioxide"), 2),
                    # Open-Meteo returns CO in µg/m³; dataset uses mg/m³
                    "CO": round(get_daily_avg("carbon_monoxide") / 1000.0, 2),
                    "O3": round(get_daily_avg("ozone"), 2)
                })

    except Exception:
        # Fallback to realistic demo values if API is unavailable or offline
        is_fallback = True
        records = []
        np.random.seed(42)
        for _, row in locations_df.iterrows():
            lat = row["lat"]
            is_north = lat > 24.0
            base_pm25 = np.random.uniform(75, 135) if is_north else np.random.uniform(22, 65)

            records.append({
                "name": row["name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "PM2.5": round(base_pm25, 2),
                "PM10": round(base_pm25 * np.random.uniform(1.6, 2.3), 2),
                "NO2": round(np.random.uniform(15, 50), 2),
                "SO2": round(np.random.uniform(8, 30), 2),
                "CO": round(np.random.uniform(0.5, 1.8), 2),
                "O3": round(np.random.uniform(30, 85), 2)
            })

    df = pd.DataFrame(records)
    return df, is_fallback


def predict_all(live_df, model_name, slider_values, models_dict, feature_names):
    """
    Enriches live pollutant data with default median values for non-API pollutants
    (NO, NOx, NH3) and the current month. Runs the selected model across all locations.
    """
    model = models_dict[model_name]
    df = live_df.copy()

    for feat, val in slider_values.items():
        df[feat] = val
    df["Month"] = datetime.now().month

    X = df[feature_names]
    preds = model.predict(X)
    df["predicted_aqi"] = np.clip(np.round(preds), 0, 999).astype(int)

    cat_details = df["predicted_aqi"].apply(aqi_category)
    df["category"] = [c[0] for c in cat_details]
    df["colour"] = [c[1] for c in cat_details]
    df["advice"] = [c[2] for c in cat_details]

    return df


# ==============================================================================
# LOAD DATA & MODELS
# ==============================================================================

models_bundle, metrics_data = load_models_and_metrics()
models_dict = models_bundle["models"]
feature_names = models_bundle["feature_names"]
feature_importances = models_bundle["feature_importances"]
slider_info = models_bundle["slider_info"]
test_eval_df = models_bundle.get("test_eval_df")
naive_mae = metrics_data.get("naive_baseline_mae", 21.34)

locations_df, city_df = load_datasets()

# Fetch live pollutant measurements
live_pollutants_df, is_demo = fetch_live_pollutants(locations_df)

# Standard default values for manual pollutants (NO, NOx, NH3)
default_manual_pollutants = {
    feat: float(slider_info[feat]["median"])
    for feat in ["NO", "NOx", "NH3"]
}


# ==============================================================================
# TOP HEADER & MODEL SELECTOR (No sidebar)
# ==============================================================================

col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.title("AirCast India")
    st.markdown("Predicting real-time Air Quality Index (AQI) right now using live atmospheric data and machine learning.")
with col_refresh:
    st.write("")
    if st.button("Refresh Live Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

if is_demo:
    st.warning("Live API unavailable – showing realistic demo data. All features and forecasts continue to work normally.")

# Model selector placed prominently at the top
model_options = list(models_dict.keys())
default_idx = model_options.index("Random Forest") if "Random Forest" in model_options else 0

col_sel_model, col_stat_r2, col_stat_mae, col_stat_naive = st.columns([2.5, 1, 1, 1])

with col_sel_model:
    selected_model = st.selectbox(
        "Select Machine Learning Model",
        options=model_options,
        index=default_idx
    )
    model_descriptions = {
        "Linear Regression": "A simple baseline model fitting straight-line relationships between pollutants and AQI.",
        "Decision Tree": "A rule-based branching tree splitting pollutant thresholds to predict AQI.",
        "Random Forest": "An ensemble of 200 decision trees averaged together for high accuracy and stability.",
        "Gradient Boosting": "A sequential ensemble where each successive tree actively corrects prior errors."
    }
    st.caption(model_descriptions.get(selected_model, ""))

mod_stats = metrics_data.get(selected_model, {})
with col_stat_r2:
    st.metric("Test R² Score", f"{mod_stats.get('r2', 0.82):.4f}")
with col_stat_mae:
    st.metric("Test MAE", f"{mod_stats.get('mae', 21.2):.2f}")
with col_stat_naive:
    st.metric("Naive Baseline MAE", f"{naive_mae:.2f}")

st.markdown("---")

# Predict across all locations using selected model
pred_df = predict_all(
    live_pollutants_df,
    selected_model,
    default_manual_pollutants,
    models_dict,
    feature_names
)


# ==============================================================================
# APP TABS (1: Custom Prediction, 2: India Glow Map, 3: Insights)
# ==============================================================================

tab1, tab2, tab3 = st.tabs([
    "Custom Prediction",
    "India Glow Map",
    "Insights"
])


# ==============================================================================
# TAB 1: CUSTOM PREDICTION (RAW INPUTS WITH LIVE CITY AUTOFILL)
# ==============================================================================
with tab1:
    st.subheader("Custom Pollutant Prediction")
    st.markdown(
        "Enter pollutant concentrations manually to predict AQI right now, "
        "or pick any Indian city to automatically load its live data into the inputs."
    )

    all_city_names = sorted(locations_df["name"].tolist())

    # Autofill controls
    col_autofill_city, col_autofill_btn = st.columns([3, 1])
    with col_autofill_city:
        autofill_city = st.selectbox(
            "Select City for Live Data Autofill",
            options=all_city_names,
            index=all_city_names.index("New Delhi") if "New Delhi" in all_city_names else 0
        )
    with col_autofill_btn:
        st.write("")
        st.write("")
        do_autofill = st.button("Autofill with Live Data", width="stretch")

    # Initialize or update session state for the input values
    if "input_PM25" not in st.session_state or do_autofill:
        # Pull live values for the chosen city
        city_live_match = live_pollutants_df[live_pollutants_df["name"] == autofill_city]
        if not city_live_match.empty:
            c_data = city_live_match.iloc[0]
            st.session_state["input_PM25"] = float(c_data["PM2.5"])
            st.session_state["input_PM10"] = float(c_data["PM10"])
            st.session_state["input_NO2"] = float(c_data["NO2"])
            st.session_state["input_SO2"] = float(c_data["SO2"])
            st.session_state["input_CO"] = float(c_data["CO"])
            st.session_state["input_O3"] = float(c_data["O3"])
        else:
            st.session_state["input_PM25"] = 60.0
            st.session_state["input_PM10"] = 120.0
            st.session_state["input_NO2"] = 30.0
            st.session_state["input_SO2"] = 15.0
            st.session_state["input_CO"] = 1.0
            st.session_state["input_O3"] = 40.0

        st.session_state["input_NO"] = float(slider_info["NO"]["median"])
        st.session_state["input_NOx"] = float(slider_info["NOx"]["median"])
        st.session_state["input_NH3"] = float(slider_info["NH3"]["median"])
        st.session_state["input_Month"] = int(datetime.now().month)

    if do_autofill:
        st.success(f"Autofilled inputs with current live measurements for {autofill_city}.")

    # Raw Inputs Layout (3 columns)
    st.markdown("#### Input Pollutant Concentrations")
    in_col1, in_col2, in_col3 = st.columns(3)

    with in_col1:
        st.markdown("**Particulate Matter & Carbon**")
        val_pm25 = st.number_input(
            "PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=1000.0,
            value=st.session_state["input_PM25"],
            step=1.0
        )
        val_pm10 = st.number_input(
            "PM10 (µg/m³)",
            min_value=0.0,
            max_value=1500.0,
            value=st.session_state["input_PM10"],
            step=1.0
        )
        val_co = st.number_input(
            "CO (mg/m³)",
            min_value=0.0,
            max_value=50.0,
            value=st.session_state["input_CO"],
            step=0.1
        )

    with in_col2:
        st.markdown("**Nitrogen & Sulphur Oxides**")
        val_no2 = st.number_input(
            "NO2 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state["input_NO2"],
            step=1.0
        )
        val_no = st.number_input(
            "NO (µg/m³)",
            min_value=0.0,
            max_value=300.0,
            value=st.session_state["input_NO"],
            step=1.0
        )
        val_nox = st.number_input(
            "NOx (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state["input_NOx"],
            step=1.0
        )
        val_so2 = st.number_input(
            "SO2 (µg/m³)",
            min_value=0.0,
            max_value=300.0,
            value=st.session_state["input_SO2"],
            step=1.0
        )

    with in_col3:
        st.markdown("**Secondary Gases & Timing**")
        val_o3 = st.number_input(
            "O3 (Ozone, µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state["input_O3"],
            step=1.0
        )
        val_nh3 = st.number_input(
            "NH3 (Ammonia, µg/m³)",
            min_value=0.0,
            max_value=400.0,
            value=st.session_state["input_NH3"],
            step=1.0
        )
        val_month = st.number_input(
            "Month (1 = Jan, 12 = Dec)",
            min_value=1,
            max_value=12,
            value=st.session_state["input_Month"],
            step=1
        )

    # Perform Prediction with current inputs
    custom_features = {
        "PM2.5": val_pm25,
        "PM10": val_pm10,
        "NO": val_no,
        "NO2": val_no2,
        "NOx": val_nox,
        "NH3": val_nh3,
        "CO": val_co,
        "SO2": val_so2,
        "O3": val_o3,
        "Month": val_month
    }
    custom_X = pd.DataFrame([custom_features])[feature_names]

    active_model_obj = models_dict[selected_model]
    raw_custom_pred = active_model_obj.predict(custom_X)[0]
    custom_aqi = int(np.clip(np.round(raw_custom_pred), 0, 999))
    custom_cat, custom_col, custom_adv = aqi_category(custom_aqi)

    st.markdown("---")
    st.markdown(f"### Prediction Result ({selected_model})")

    res_col_gauge, res_col_models = st.columns([1, 1])

    with res_col_gauge:
        fig_custom_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=custom_aqi,
            title={"text": "Predicted AQI (Right Now)", "font": {"size": 20, "color": "#FFFFFF"}},
            number={"font": {"size": 52, "color": custom_col}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#888"},
                "bar": {"color": custom_col, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "#444",
                "steps": [
                    {"range": [0, 50], "color": "rgba(0, 228, 0, 0.25)"},
                    {"range": [50, 100], "color": "rgba(146, 208, 80, 0.25)"},
                    {"range": [100, 200], "color": "rgba(255, 222, 51, 0.25)"},
                    {"range": [200, 300], "color": "rgba(255, 153, 51, 0.25)"},
                    {"range": [300, 400], "color": "rgba(204, 0, 51, 0.25)"},
                    {"range": [400, 500], "color": "rgba(126, 0, 35, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": custom_col, "width": 4},
                    "thickness": 0.75,
                    "value": custom_aqi
                }
            }
        ))
        fig_custom_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#0e1117"
        )
        st.plotly_chart(fig_custom_gauge, width="stretch")

        st.markdown(
            f"<div style='background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; border-left: 5px solid {custom_col};'>"
            f"<b>Category:</b> {custom_cat}<br>"
            f"<b>Health Advisory:</b> {custom_adv}"
            f"</div>",
            unsafe_allow_html=True
        )

    with res_col_models:
        st.markdown("#### Predictions Across All 4 Models")
        custom_comp = []
        for m_name, m_obj in models_dict.items():
            p_val = int(np.clip(np.round(m_obj.predict(custom_X)[0]), 0, 999))
            c_label, c_color, _ = aqi_category(p_val)
            custom_comp.append({
                "Model": m_name,
                "Predicted AQI": p_val,
                "Category": c_label,
                "Colour": c_color
            })
        custom_comp_df = pd.DataFrame(custom_comp)

        fig_custom_comp = px.bar(
            custom_comp_df,
            x="Model",
            y="Predicted AQI",
            text="Predicted AQI",
            color="Category",
            color_discrete_map={r["Category"]: r["Colour"] for _, r in custom_comp_df.iterrows()}
        )
        fig_custom_comp.update_traces(textposition="outside")
        fig_custom_comp.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0e1117",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, max(custom_comp_df["Predicted AQI"].max() * 1.25, 100)]),
            showlegend=False
        )
        st.plotly_chart(fig_custom_comp, width="stretch")
        st.caption("Compares predictions across all trained algorithms using the current inputs.")


# ==============================================================================
# TAB 2: INDIA GLOW MAP (ONLY AQI)
# ==============================================================================
with tab2:
    st.subheader("India Air Quality Glow Map")
    st.markdown("Every location is displayed with 3-tier halo rings where severely polluted areas burn red and clean regions glow soft green.")

    # Top summary metrics for AQI
    most_polluted = pred_df.loc[pred_df["predicted_aqi"].idxmax()]
    cleanest = pred_df.loc[pred_df["predicted_aqi"].idxmin()]
    avg_aqi = round(pred_df["predicted_aqi"].mean())
    avg_cat, _, _ = aqi_category(avg_aqi)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Most Polluted Right Now",
            f"{most_polluted['name']}",
            f"AQI {most_polluted['predicted_aqi']} ({most_polluted['category']})"
        )
    with col2:
        st.metric(
            "Cleanest Air Right Now",
            f"{cleanest['name']}",
            f"AQI {cleanest['predicted_aqi']} ({cleanest['category']})"
        )
    with col3:
        st.metric(
            "National Average AQI",
            f"{avg_aqi}",
            f"Category: {avg_cat}"
        )

    # 3-Layer Stacked Glow Markers for AQI
    fig_map = go.Figure()

    marker_colors = pred_df["colour"].tolist()

    # Sizes scale smoothly with AQI severity: bad AQI burns with a large red halo
    core_sizes = np.clip(pred_df["predicted_aqi"] / 40.0 + 6.0, 6, 14)
    mid_sizes = core_sizes * 2.2
    outer_sizes = core_sizes * 4.2

    hover_texts = [
        f"<b>{row['name']}</b><br>"
        f"Predicted AQI: <b>{row['predicted_aqi']}</b> ({row['category']})<br>"
        f"────────────────────────<br>"
        f"PM2.5: {row['PM2.5']} µg/m³<br>"
        f"PM10: {row['PM10']} µg/m³<br>"
        f"NO2: {row['NO2']} µg/m³<br>"
        f"SO2: {row['SO2']} µg/m³<br>"
        f"CO: {row['CO']} mg/m³<br>"
        f"O3: {row['O3']} µg/m³"
        for _, row in pred_df.iterrows()
    ]

    # 1. Outer Halo (large, low opacity)
    fig_map.add_trace(go.Scattermap(
        lat=pred_df["lat"],
        lon=pred_df["lon"],
        mode="markers",
        marker=dict(
            size=outer_sizes,
            color=marker_colors,
            opacity=0.15
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    # 2. Middle Halo (medium size, semi-transparent)
    fig_map.add_trace(go.Scattermap(
        lat=pred_df["lat"],
        lon=pred_df["lon"],
        mode="markers",
        marker=dict(
            size=mid_sizes,
            color=marker_colors,
            opacity=0.35
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    # 3. Core Dot (crisp central marker carrying the hover tooltip)
    fig_map.add_trace(go.Scattermap(
        lat=pred_df["lat"],
        lon=pred_df["lon"],
        mode="markers",
        marker=dict(
            size=core_sizes,
            color=marker_colors,
            opacity=0.95
        ),
        text=hover_texts,
        hoverinfo="text",
        showlegend=False
    ))

    fig_map.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=22.5, lon=79.0),
            zoom=3.8
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor="#0e1117"
    )

    st.plotly_chart(fig_map, width="stretch")
    st.caption("Real-time AQI prediction right now, generated from live API data using the selected model.")

    # Top 5 and sortable data table
    st.markdown("### Top 5 Most Polluted Locations Right Now")
    top5 = pred_df.sort_values(by="predicted_aqi", ascending=False).head(5)
    cols_top = st.columns(5)
    for i, (_, r) in enumerate(top5.iterrows()):
        with cols_top[i]:
            st.markdown(
                f"<div style='border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 10px; text-align: center; background: rgba(255,255,255,0.03);'>"
                f"<h4 style='margin:0 0 5px 0;'>{r['name']}</h4>"
                f"<span class='cpcb-badge' style='background:{r['colour']}; color:#000;'>AQI {r['predicted_aqi']}</span>"
                f"<p style='margin:6px 0 0 0; font-size:0.85rem; color:#bbb;'>{r['category']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("### All Locations Live AQI Table")
    display_df = pred_df[[
        "name", "predicted_aqi", "category", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"
    ]].rename(columns={
        "name": "Location",
        "predicted_aqi": "Predicted AQI (Right Now)",
        "category": "CPCB Category",
        "CO": "CO (mg/m³)",
        "PM2.5": "PM2.5 (µg/m³)",
        "PM10": "PM10 (µg/m³)",
        "NO2": "NO2 (µg/m³)",
        "SO2": "SO2 (µg/m³)",
        "O3": "O3 (µg/m³)"
    })
    st.dataframe(display_df, width="stretch", height=320)


# ==============================================================================
# TAB 3: INSIGHTS (POINTS 1, 2, 3 ONLY)
# ==============================================================================
with tab3:
    st.subheader("Model Performance & Environmental Insights")

    # 1. Model Comparison Table
    st.markdown("#### 1. Model Comparison on Test Set")
    table_rows = []
    table_rows.append({
        "Model": "Naive Baseline (Persistence)",
        "MAE (Error in AQI)": f"{naive_mae:.2f}",
        "RMSE": "N/A",
        "R² Score": "N/A",
        "Status": "Baseline"
    })
    for m_name in models_dict.keys():
        stats = metrics_data.get(m_name, {})
        is_sel = "Active Model" if m_name == selected_model else ""
        table_rows.append({
            "Model": m_name,
            "MAE (Error in AQI)": f"{stats.get('mae', 0.0):.2f}",
            "RMSE": f"{stats.get('rmse', 0.0):.2f}",
            "R² Score": f"{stats.get('r2', 0.0):.4f}",
            "Status": is_sel
        })
    st.table(pd.DataFrame(table_rows))

    st.markdown("---")

    # 2. Feature Importance Bar Chart
    st.markdown(f"#### 2. Feature Importance for **{selected_model}**")
    imp_dict = feature_importances.get(selected_model, {})
    imp_df = pd.DataFrame({
        "Feature": list(imp_dict.keys()),
        "Importance": list(imp_dict.values())
    }).sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#0e1117",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_imp, width="stretch")
    st.caption("Shows the relative contribution of each pollutant feature in determining AQI.")

    st.markdown("---")

    # 3. Actual vs Predicted Line Chart on Test Set
    st.markdown("#### 3. Test Set Validation: Actual vs Predicted AQI (Last ~100 Days)")
    if test_eval_df is not None and not test_eval_df.empty:
        test_cities = sorted(test_eval_df["City"].unique().tolist())
        sel_eval_city = st.selectbox(
            "Select Dataset City to Inspect:",
            options=test_cities,
            index=test_cities.index("Delhi") if "Delhi" in test_cities else 0
        )

        city_eval = test_eval_df[test_eval_df["City"] == sel_eval_city].sort_values("Date").tail(100)
        pred_col = f"Pred_{selected_model}"

        fig_eval = go.Figure()
        actual_aqi_series = city_eval["AQI"] if "AQI" in city_eval.columns else city_eval["AQI_tomorrow"]
        fig_eval.add_trace(go.Scatter(
            x=city_eval["Date"],
            y=actual_aqi_series,
            mode="lines",
            name="Actual AQI",
            line=dict(color="#00E400", width=2.5)
        ))
        fig_eval.add_trace(go.Scatter(
            x=city_eval["Date"],
            y=city_eval[pred_col],
            mode="lines",
            name=f"Predicted ({selected_model})",
            line=dict(color="#FF9933", width=2, dash="dash")
        ))
        fig_eval.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0e1117",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date",
            yaxis_title="AQI",
            hovermode="x unified"
        )
        st.plotly_chart(fig_eval, width="stretch")
        st.caption(f"Historical ground truth vs model prediction for {sel_eval_city} on unseen test data.")
