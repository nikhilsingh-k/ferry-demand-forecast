import pandas as pd
import numpy as np
from typing import Dict


def compute_kpis(
    y_true: pd.Series,
    y_pred: pd.Series,
    intervals: pd.DataFrame
) -> Dict:
    """
    Compute business-relevant KPIs for forecasting system.
    """

    # ======================================================
    # ALIGNMENT (CRITICAL)
    # ======================================================
    y_true, y_pred = y_true.align(y_pred, join="inner")
    intervals = intervals.loc[y_true.index]

    # Ensure float
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    # ======================================================
    # ERRORS
    # ======================================================
    errors = y_true - y_pred
    abs_errors = errors.abs()

    # ======================================================
    # 1. FORECAST ACCURACY (MAPE-based)
    # ======================================================
    mask = y_true != 0
    if mask.sum() > 0:
        mape = (abs_errors[mask] / y_true[mask]).mean() * 100
    else:
        mape = np.nan

    accuracy = 100 - mape if not np.isnan(mape) else np.nan

    # ======================================================
    # 2. ERROR DRIFT (TIME-AWARE)
    # ======================================================
    midpoint = len(abs_errors) // 2

    first_half = abs_errors.iloc[:midpoint]
    second_half = abs_errors.iloc[midpoint:]

    first_mae = first_half.mean() if len(first_half) > 0 else 0
    second_mae = second_half.mean() if len(second_half) > 0 else 0

    error_drift = second_mae - first_mae

    # ======================================================
    # 3. PEAK MISS RATE
    # ======================================================
    peak_threshold = y_true.quantile(0.9)
    peak_mask = y_true >= peak_threshold

    if peak_mask.sum() > 0:
        peak_errors = abs_errors[peak_mask]

        dynamic_threshold = np.maximum(
            0.20 * y_true[peak_mask],
            5
        )

        missed = peak_errors > dynamic_threshold
        peak_miss_rate = (missed.sum() / peak_mask.sum()) * 100
    else:
        peak_miss_rate = 0.0

    # ======================================================
    # 4. CONFIDENCE BAND WIDTH
    # ======================================================
    if {"upper_bound", "lower_bound"}.issubset(intervals.columns):
        band_width = intervals["upper_bound"] - intervals["lower_bound"]
        avg_band_width = band_width.mean()
    else:
        avg_band_width = np.nan

    # ======================================================
    # 5. FORECAST LEAD TIME
    # ======================================================
    forecast_lead_time = "TODO"

    # ======================================================
    # FINAL OUTPUT
    # ======================================================
    kpis = {
        "Forecast_Accuracy": round(float(accuracy), 2) if not np.isnan(accuracy) else None,
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else None,
        "Error_Drift": round(float(error_drift), 4),
        "Peak_Miss_Rate": round(float(peak_miss_rate), 2),
        "Confidence_Band_Width": round(float(avg_band_width), 2) if not np.isnan(avg_band_width) else None,
        "Forecast_Lead_Time": forecast_lead_time,
    }

    print("✅ KPIs computed")
    for k, v in kpis.items():
        print(f"{k}: {v}")

    return kpis
