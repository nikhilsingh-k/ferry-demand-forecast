import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Pipeline imports
from src.data_loader import load_ferry_data
from src.features import create_features
from src.train_test_split import time_split
from src.baseline_models import (
    naive_forecast,
    moving_average_forecast,
    linear_regression_forecast
)
from src.evaluation import evaluate_all
from src.uncertainty import calculate_prediction_intervals, get_train_residuals
from src.kpis import compute_kpis

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="⛴️",
    layout="wide"
)

st.title("⛴️ Ferry Demand Forecasting Dashboard")
st.markdown("**Short-Term Ticket Demand Prediction (15-min intervals)**")

# ====================== SIDEBAR ======================
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    options=["Naive", "Moving Average (4)", "Linear Regression"],
    index=2
)

train_ratio = st.sidebar.slider(
    "Train/Test Split Ratio",
    min_value=0.6,
    max_value=0.95,
    value=0.80,
    step=0.05
)

run_button = st.sidebar.button("🚀 Run Forecasting Pipeline", type="primary")

# ====================== HELPERS ======================
def safe_metric(value, suffix=""):
    return f"{value}{suffix}" if value is not None else "N/A"

# ====================== MAIN ======================
if run_button:
    with st.spinner("Running full forecasting pipeline..."):
        try:
            # 1. Load Data (robust path)
            DATA_PATH = Path("data/Toronto Island Ferry Tickets.csv")
            df_raw = load_ferry_data(DATA_PATH)

            # 2. Feature Engineering
            df_features = create_features(df_raw)

            # 3. Time Split
            X_train, X_test, y_train, y_test = time_split(
                df_features, train_ratio=train_ratio
            )

            # 4. Model Prediction
            if model_choice == "Naive":
                y_pred = naive_forecast(y_train, y_test.index)
                model_name = "Naive Forecast"

            elif model_choice == "Moving Average (4)":
                y_pred = moving_average_forecast(
                    y_train, y_test.index, window=4
                )
                model_name = "Moving Average (Window=4)"

            else:
                y_pred = linear_regression_forecast(
                    X_train, y_train, X_test
                )
                model_name = "Linear Regression"

            # 5. Evaluation
            metrics = evaluate_all(y_test, y_pred)

            # 6. Proper Residuals (NO FAKE LOGIC)
            if model_choice == "Linear Regression":
                y_train_pred = linear_regression_forecast(
                    X_train, y_train, X_train
                )

            elif model_choice == "Naive":
                y_train_pred = naive_forecast(y_train, y_train.index)

            else:
                y_train_pred = moving_average_forecast(
                    y_train, y_train.index, window=4
                )

            residuals = get_train_residuals(y_train, y_train_pred)

            # 7. Prediction Intervals
            intervals = calculate_prediction_intervals(y_pred, residuals)

            # 8. KPIs
            kpis = compute_kpis(y_test, y_pred, intervals)

            # ====================== OUTPUT ======================
            st.success(f"✅ Pipeline completed using **{model_name}**")

            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Forecast Accuracy",
                    safe_metric(kpis["Forecast_Accuracy"], "%")
                )

            with col2:
                st.metric(
                    "Peak Miss Rate",
                    safe_metric(kpis["Peak_Miss_Rate"], "%")
                )

            with col3:
                st.metric(
                    "Error Drift",
                    safe_metric(kpis["Error_Drift"])
                )

            with col4:
                st.metric(
                    "Confidence Band Width",
                    safe_metric(kpis["Confidence_Band_Width"])
                )

            # ====================== PLOT ======================
            st.subheader("Actual vs Predicted Demand")

            fig = go.Figure()

            # Actual
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=y_test,
                mode="lines",
                name="Actual",
            ))

            # Predicted
            fig.add_trace(go.Scatter(
                x=y_pred.index,
                y=y_pred,
                mode="lines",
                name=f"Predicted ({model_name})",
            ))

            # Upper bound
            fig.add_trace(go.Scatter(
                x=intervals.index,
                y=intervals["upper_bound"],
                line=dict(width=0),
                showlegend=False
            ))

            # Lower bound + fill
            fig.add_trace(go.Scatter(
                x=intervals.index,
                y=intervals["lower_bound"],
                fill="tonexty",
                name="95% Interval"
            ))

            fig.update_layout(
                height=600,
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Sales Count"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ====================== ERROR PLOT (BONUS) ======================
            st.subheader("Forecast Error Over Time")
            error_series = y_test - y_pred
            st.line_chart(error_series)

            # ====================== METRICS ======================
            st.subheader("Detailed Metrics")

            colA, colB = st.columns(2)

            with colA:
                st.write("**Error Metrics**")
                st.json(metrics)

            with colB:
                st.write("**Business KPIs**")
                st.json(kpis)

            # ====================== DATA ======================
            with st.expander("View Predictions Table"):
                df_compare = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": y_pred,
                    "Lower": intervals["lower_bound"],
                    "Upper": intervals["upper_bound"]
                })
                st.dataframe(df_compare.head(50))

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Check dataset path and module imports.")

else:
    st.info("👈 Configure settings in sidebar and click Run")

    st.markdown("""
    ### Instructions:
    1. Place dataset in `data/` folder  
    2. Select model  
    3. Run pipeline  
    """)

# Footer
st.caption("End-to-End Forecasting System with Uncertainty & KPIs")
