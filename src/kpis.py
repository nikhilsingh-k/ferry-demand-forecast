import pandas as pd
import numpy as np


def compute_kpis(y_true, y_pred, intervals):

    y_true, y_pred = y_true.align(y_pred, join="inner")

    errors = (y_true - y_pred).abs()

    # Forecast Accuracy ----------------------------------------------------------------------
 
    epsilon = 20
    mape = (errors / (y_true.abs() + epsilon)).mean() * 100
    mape = np.clip(mape, 0, 85)
    forecast_accuracy = max(0, 100 - mape)


    # PEAK MISS RATE -----------------------------------------------------------------

    # Ferry data has extreme outliers (max=5342, mean=40).
    # Using Q90 (=122) as threshold means any spike > 122 is a "peak",
    q25 = float(y_true.quantile(0.25))
    q50 = float(y_true.quantile(0.50))
    q75 = float(y_true.quantile(0.75))
    iqr  = q75 - q25

    robust_threshold = q50 + 2 * iqr          # median + 2*IQR
    q70 = float(y_true.quantile(0.70))
    q85 = float(y_true.quantile(0.85))
    peak_threshold = float(np.clip(robust_threshold, q70, q85))

    actual_peak_idx = np.where(y_true.values >= peak_threshold)[0]

    # Window = ±4 steps (±60 min) — realistic for ferry scheduling
    missed = 0
    for idx in actual_peak_idx:
        window_start = max(0, idx - 4)
        window_end   = min(len(y_pred), idx + 5)
        window_max   = y_pred.iloc[window_start:window_end].max()
        # A "hit" = predicted value reaches at least 60% of the peak threshold
        # This accounts for models that underestimate extreme spikes
        if window_max < peak_threshold * 0.60:
            missed += 1

    if len(actual_peak_idx) > 0:
        peak_miss_rate = (missed / len(actual_peak_idx)) * 100
    else:
        peak_miss_rate = 0.0

    # Error Drift ----------------------------------------------------------
  
    n = len(errors)
    half = n // 2
    mae_first  = float(errors.iloc[:half].mean()) if half > 0 else 0.0
    mae_second = float(errors.iloc[half:].mean()) if half > 0 else 0.0
    error_drift = mae_second - mae_first

 
    # Confidence Band Width ------------------------------------------------------

    band_width = (
        intervals["upper_bound"] - intervals["lower_bound"]
    ).mean()


    # FORECAST LEAD TIME ----------------------------------------------------------

    lead_time = np.nan
    try:
        actual_peaks_s = y_true[y_true >= peak_threshold]

        pred_index_plain = pd.DatetimeIndex(y_pred.index.values)
        pred_vals        = np.asarray(y_pred.values, dtype=float)
        # Lead time uses the same 60% threshold for consistency
        pred_peaks_mask  = pred_vals >= peak_threshold * 0.60
        pred_peaks_index = pred_index_plain[pred_peaks_mask]

        lead_times = []
        max_lead   = pd.Timedelta(minutes=120)

        for actual_time in actual_peaks_s.index:
            early_preds = pred_peaks_index[
                (pred_peaks_index <  actual_time) &
                (pred_peaks_index >= actual_time - max_lead)
            ]
            if len(early_preds) > 0:
                nearest_pred = early_preds.max()
                lead_minutes = (actual_time - nearest_pred).total_seconds() / 60
                if lead_minutes >= 15:
                    lead_times.append(lead_minutes)

        if len(lead_times) > 0:
            lead_time = round(float(np.mean(lead_times)), 0)

    except Exception:
        lead_time = np.nan

    return {
        "Forecast_Accuracy":         round(float(forecast_accuracy), 2),
        "Peak_Miss_Rate":            round(float(peak_miss_rate),    2),
        "Error_Drift":               round(float(error_drift),       3),
        "Confidence_Band_Width":     round(float(band_width),        2),
        "Forecast_Lead_Time (mins)": (
            round(float(lead_time), 0)
            if lead_time is not None and not pd.isna(lead_time)
            else 0
        ),
    }
