import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.rolling_validation import rolling_forecast_validation
from src.data_loader import load_ferry_data
from src.features import create_features
from src.train_test_split import time_split
from src.baseline_models import (
    naive_forecast, moving_average_forecast,
    linear_regression_forecast, random_forest_forecast,
    gradient_boosting_forecast, xgboost_forecast,
)
from src.time_series_models import arima_forecast
from src.prophet_model import prophet_forecast
from src.evaluation import evaluate_all
from src.uncertainty import calculate_prediction_intervals, get_train_residuals
from src.kpis import compute_kpis
from src.horizon_metrics import horizon_metrics

st.set_page_config(
    page_title="Ferry Demand Forecasting",
    layout="wide",
    page_icon="⛴️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0A0F1C;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Header */
    .header-card {
        background: linear-gradient(135deg, #1E2937 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* KPI Cards */
    .kpi-card {
        background-color: #1E2937;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155;
        color: #FACC15;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #FACC15, #F59E0B);
        color: #0F172A;
        font-weight: 700;
        border-radius: 12px;
        height: 52px;
        font-size: 17px;
        border: none;
    }
    
    /* Plotly Charts */
    .js-plotly-plot .plotly .modebar {
        background-color: #1E2937 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h1 style='color:#FACC15; margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-1px;'>
        ⛴️ Ferry Demand Forecasting
    </h1>
    <p style='color:#94A3B8; font-size:1.1rem; margin:8px 0 0 0;'>
        Toronto Island Park • Real-time 15-minute Demand Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

HORIZON_MAP = {"15 min": 1, "30 min": 2, "1 hour": 4, "2 hour": 8}

MODEL_MAP = {
    "Naive": naive_forecast,
    "Moving Average": moving_average_forecast,
    "Linear Regression": linear_regression_forecast,
    "Random Forest": random_forest_forecast,
    "Gradient Boosting": gradient_boosting_forecast,
    "XGBoost": xgboost_forecast,
}

with st.sidebar:
    st.sidebar.markdown("## ⚙️ Forecast Controls")
    st.sidebar.markdown("---")
    model_choice = st.selectbox("Primary Model", list(MODEL_MAP.keys()) + ["ARIMA", "Prophet"], index=2)
    use_rolling = st.checkbox("Use Rolling Validation (Advanced)", value=False)
    compare_all = st.checkbox("Compare All ML Models", value=False)
    if compare_all:
        st.warning("⚠️ Running all models on large historical data may take 30–60 seconds. This ensures better accuracy and full comparison.")
    horizon_choice = st.selectbox("Forecast Horizon", list(HORIZON_MAP.keys()), index=2)
    horizon_steps = HORIZON_MAP[horizon_choice]
    train_ratio = st.slider("Train/Test Split Ratio", 0.60, 0.95, 0.80, 0.05)
    run_button = st.button("🚀 RUN FORECAST", type="primary", use_container_width=True)



if not run_button:
    st.info("👈 Configure sidebar and click **RUN FORECAST**")
    st.stop()

with st.spinner("Loading data..."):
    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_feat = create_features(df_raw.copy())
    X_train, X_test, y_train, y_test = time_split(df_feat, train_ratio)
    selected_time = st.selectbox(
    "🕒 Select Forecast Start Time",
    options=X_test.index.astype(str),
    index=0
    )

    selected_time = pd.to_datetime(selected_time)
    # ==================== PERFORMANCE FIX ====================
    MAX_TRAIN = 50000
    MAX_TEST = 2000

    if len(X_train) > MAX_TRAIN:
        X_train = X_train.iloc[-MAX_TRAIN:]
        y_train = y_train.iloc[-MAX_TRAIN:]

    if len(X_test) > MAX_TEST:
        X_test = X_test.iloc[:MAX_TEST]
        y_test = y_test.iloc[:MAX_TEST]

# ==================== ROLLING VALIDATION ====================
if use_rolling:
    st.subheader("🔄 Rolling Validation Results")

    model_func = MODEL_MAP.get(model_choice)

    if model_func is None:
        st.warning("⚠️ Rolling validation only supports ML models (not ARIMA/Prophet)")
        st.stop()

    roll_df = rolling_forecast_validation(
        model_func,
        X_train,
        y_train,
        initial_train_size=30000,
        step_size=5000,
        horizon=horizon_steps
    )

    if not roll_df.empty:
        st.dataframe(roll_df, use_container_width=True)
        st.line_chart(roll_df.set_index("test_start")["MAE"])
    else:
        st.warning("No rolling validation results generated.")

    st.stop()

# ==================== Apply Horizon Correctly ====================
X_test_full = X_test.copy()
y_test_full = y_test.copy()


forecast_end = selected_time + pd.Timedelta(minutes=15 * horizon_steps)

mask = (X_test_full.index >= selected_time) & (X_test_full.index <= forecast_end)

X_test = X_test_full.loc[mask].copy()
y_test = y_test_full.loc[mask].copy()

#full dataset
X_eval = X_test_full
y_eval = y_test_full

# Safety fallback
if len(X_test) < 8:
    X_test = X_test_full.iloc[:32]
    y_test = y_test_full.iloc[:32]

models_to_run = list(MODEL_MAP.keys()) if compare_all else [model_choice]
predictions = {}

# Temporary message placeholder
loading_msg = st.empty()

loading_msg.info(f"⏳ Running {len(models_to_run)} model(s)... please wait")

with st.spinner(f"Running {len(models_to_run)} model(s)..."):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, name in enumerate(models_to_run):
        status_text.text(f"Running model: {name}...")

        try:
            if name in MODEL_MAP:
                pred = MODEL_MAP[name](X_train, y_train, X_test)

            elif name == "ARIMA":
                raw = arima_forecast(y_train.iloc[-5000:], len(X_test))
                pred = pd.Series(
                    raw.values if hasattr(raw, 'values') else raw,
                    index=X_test.index,
                    name="ARIMA"
                )

            elif name == "Prophet":
                raw = prophet_forecast(df_feat.iloc[-5000:], len(X_test))
                pred = pd.Series(
                    raw.values if hasattr(raw, 'values') else raw,
                    index=X_test.index,
                    name="Prophet"
                )

            else:
                pred = pd.Series([0] * len(X_test), index=X_test.index)

            if pred.isna().all():
                raise ValueError("Model returned all NaN predictions")

            pred = pred.reindex(X_test.index).ffill().bfill().clip(lower=0)
            predictions[name] = pred

        except Exception as e:
            st.error(f"{name} FAILED: {str(e)}")

        progress_bar.progress((i + 1) / len(models_to_run))

# CLEANUP AFTER FINISH
progress_bar.empty()
status_text.empty()
loading_msg.empty()

if not predictions:
    st.error("All models failed.")
    st.stop()

primary_name = list(predictions.keys())[0]
primary_pred = predictions[primary_name]


# KPI EVALUATION


# Use ONLY selected forecast horizon
eval_pred = primary_pred.reindex(y_test.index).ffill().bfill()

# Evaluate correctly
metrics = evaluate_all(y_eval, eval_pred.reindex(y_eval.index).ffill().bfill())

# ==================== Model Residuals ====================
try:
    if primary_name in MODEL_MAP:
        train_pred = MODEL_MAP[primary_name](X_train, y_train, X_train)
        train_pred = train_pred + np.random.normal(0, y_train.std() * 0.05, size=len(train_pred))
    elif primary_name in ["ARIMA", "Prophet"]:
        train_pred = y_train.shift(1).fillna(y_train.mean())
    else:
        train_pred = linear_regression_forecast(X_train, y_train, X_train)

    residuals = get_train_residuals(y_train, train_pred)

except:
    residuals = y_train - y_train.shift(1).fillna(y_train.mean())


eval_pred_full = primary_pred.reindex(y_eval.index).ffill().bfill()

intervals = calculate_prediction_intervals(eval_pred_full, residuals)

kpis = compute_kpis(y_eval, eval_pred_full, intervals)

# ==================== Safe KPI Display ====================
def safe_format(val, fmt="{:.1f}", suffix=""):
    if val is None or pd.isna(val) or np.isinf(val):
        return "N/A"
    return fmt.format(val) + suffix

# Enhanced KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)

kpi_data = [
    {
        "label": "Forecast Accuracy",
        "value": f"{kpis.get('Forecast_Accuracy', 0):.1f}%",
        "color": "#FACC15"
    },
    {
        "label": "Peak Miss Rate",
        "value": f"{kpis.get('Peak_Miss_Rate', 0):.1f}%",
        "color": "#F87171"
    },
    {
        "label": "Error Drift",
        "value": f"{kpis.get('Error_Drift', 0):+.3f}",
        "color": "#E2E8F0"
    },
    {
        "label": "Conf. Band Width",
        "value": f"{kpis.get('Confidence_Band_Width', 0):.1f}",
        "color": "#60A5FA"
    },
    {
        "label": "Lead Time",
        "value": f"{kpis.get('Forecast_Lead_Time (mins)', 0):.0f} min",
        "color": "#34D399"
    },
]

cols = st.columns(5)
for col, kpi in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div style="
            background:#1E2937;
            border-radius:16px;
            padding:20px 12px;
            text-align:center;
            border:1px solid rgba(148,163,184,0.15);
            min-height:110px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
        ">
            <div style="color:#94A3B8; font-size:0.78rem; margin-bottom:8px; white-space:nowrap;">
                {kpi['label']}
            </div>
            <div style="font-size:1.9rem; font-weight:700; color:{kpi['color']}; white-space:nowrap; line-height:1.2;">
                {kpi['value']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Overview", "🏆 Model Comparison", "⏳ Horizon Analysis",
    "📉 Error Analysis", "🔥 Peak Analysis"
])

with tab1:
    st.markdown('<div class="section-title">Actual vs Predicted Demand</div>', unsafe_allow_html=True)
    N = min(400, len(y_test))
    y_disp = y_test.iloc[-N:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_disp.index, y=y_disp, name="Actual", line=dict(color="#e2e8f0", width=2)))
    for i, (name, pred) in enumerate(predictions.items()):
        p_disp = pred.reindex(y_disp.index).fillna(0)
        fig.add_trace(go.Scatter(x=p_disp.index, y=p_disp, name=name,
                                 line=dict(color=["#38bdf8","#facc15","#34d399"][i%3], width=1.8)))
    ub = intervals["upper_bound"].reindex(y_disp.index)
    lb = intervals["lower_bound"].reindex(y_disp.index)
    fig.add_trace(go.Scatter(x=ub.index, y=ub, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=lb.index, y=lb, fill="tonexty", name="95% CI",
                             fillcolor="rgba(56,189,248,0.2)", line=dict(width=0)))
    fig.update_layout(
        paper_bgcolor="#0B1324",
        plot_bgcolor="#0B1324",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        title=dict(
            text="Actual vs Predicted Demand",
            font=dict(color="#FACC15", size=18, family="Inter, sans-serif"),
            x=0.01
        ),
        legend=dict(
            bgcolor="rgba(30,41,55,0.8)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(color="#E2E8F0")
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.15)",
            tickfont=dict(color="#64748B"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.15)",
            tickfont=dict(color="#64748B"),
            showgrid=True,
            zeroline=False,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
        hovermode="x unified",
    )

    # Improve trace styles
    fig.data[0].update(line=dict(color="#E2E8F0", width=2), name="Actual")
    colors = ["#38BDF8", "#FACC15", "#34D399", "#F87171", "#A78BFA"]
    for i in range(1, len(fig.data) - 2):  # skip last 2 (CI band traces)
        fig.data[i].update(line=dict(width=2, color=colors[(i - 1) % len(colors)]))

    st.plotly_chart(fig, use_container_width=True)
    # Download forecast as CSV
    forecast_df = pd.DataFrame({
        "Timestamp": y_test.index,
        "Actual": y_test.values,
        "Predicted": primary_pred.reindex(y_test.index).values,
        "Upper_Bound": intervals["upper_bound"].reindex(y_test.index).values,
        "Lower_Bound": intervals["lower_bound"].reindex(y_test.index).values,
    })

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"ferry_forecast_{selected_time.date()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab2:
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    comp = []
    for name, pred in predictions.items():
        m = evaluate_all(y_test, pred)
        comp.append({
            "Model": name,
            "MAE": round(m.get("MAE", 0), 3),
            "RMSE": round(m.get("RMSE", 0), 3),
            "MAPE": round(m.get("MAPE", 0), 2),
            "Accuracy %": round(100 - m.get("MAPE", 100), 2)
        })
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-title">Horizon-wise Performance</div>', unsafe_allow_html=True)
    h_metrics = horizon_metrics(y_test, primary_pred)
    if h_metrics:
        st.dataframe(pd.DataFrame.from_dict(h_metrics, orient='index'), use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Error Analysis</div>', unsafe_allow_html=True)
    error_series = (y_test - primary_pred.reindex(y_test.index)).abs().rolling(8, min_periods=1).mean()
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=error_series.index, y=error_series, name="Rolling MAE (8 steps)"))
    fig_err.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_err, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">Peak Analysis (Top 10% Demand)</div>', unsafe_allow_html=True)
    thr = float(y_test.quantile(0.90))
    peak_mask_actual = y_test >= thr
    peak_mask_pred = primary_pred.reindex(y_test.index) >= thr
    missed_peaks = peak_mask_actual & ~peak_mask_pred
    fig_pk = go.Figure()
    fig_pk.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Actual", line=dict(color="#e2e8f0", width=1.5)))
    fig_pk.add_trace(go.Scatter(x=y_test[peak_mask_actual].index, y=y_test[peak_mask_actual],
                                mode="markers", name="Actual Peaks", marker=dict(color="#34d399", size=7)))
    fig_pk.add_trace(go.Scatter(x=y_test[peak_mask_pred].index, y=y_test[peak_mask_pred],
                                mode="markers", name="Predicted Peaks", marker=dict(color="#facc15", size=7)))
    fig_pk.add_trace(go.Scatter(x=y_test[missed_peaks].index, y=y_test[missed_peaks],
                                mode="markers", name="Missed Peaks", marker=dict(color="#f87171", size=8, symbol="x")))
    fig_pk.update_layout(height=420, template="plotly_dark")
    st.plotly_chart(fig_pk, use_container_width=True)

st.caption("Toronto Island Ferry Demand Forecasting System • Production Ready")

st.markdown("---")

st.caption("Built by AI/ML Forecasting Developer • [LinkedIn](https://www.linkedin.com/in/nikhilsingh-k/) • [GitHub](https://github.com/nikhilsingh-k)")
