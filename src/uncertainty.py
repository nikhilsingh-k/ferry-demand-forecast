import pandas as pd
import numpy as np



# PREDICTION INTERVALS --------------------------------------------------------------------

def calculate_prediction_intervals(
    y_pred: pd.Series,
    residuals: pd.Series,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute prediction intervals using residual standard deviation.

    Assumption:
    - Residuals ~ Normal distribution

    95% interval → z = 1.96
    """

    # VALIDATION --------------------------------------------------------------
    
    if len(residuals) == 0:
        raise ValueError("Residuals cannot be empty")

    # Aligned residuals (safety)
    residuals = residuals.dropna()


    # Z-SCORE (FIXED)
    if confidence_level == 0.95:
        z = 1.96
    else:
        # TODO: Generalize for other confidence levels if needed
        z = 1.96

    # STANDARD DEVIATION --------------------------------------------------------------
   
    std_residuals = residuals.std(ddof=1)

    if np.isnan(std_residuals) or std_residuals == 0:
        std_residuals = 1e-6

    # INTERVALS --------------------------------------------------------------------
    
    lower_bound = y_pred - z * std_residuals
    upper_bound = y_pred + z * std_residuals

    intervals_df = pd.DataFrame({
        "prediction": y_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }, index=y_pred.index)

    print("✅ Prediction intervals computed")
    print(f"Std residual: {std_residuals:.4f}")

    return intervals_df


# TRAIN RESIDUALS -----------------------------------------------------------------------

def get_train_residuals(
    y_true: pd.Series,
    y_pred: pd.Series
) -> pd.Series:
    """
    Compute residuals safely with index alignment
    """

    y_true, y_pred = y_true.align(y_pred, join="inner")

    residuals = y_true - y_pred

    return residuals


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_ferry_data
    from src.features import create_features
    from src.train_test_split import time_split
    from src.baseline_models import linear_regression_forecast

    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_features = create_features(df_raw)

    X_train, X_test, y_train, y_test = time_split(df_features)

    # Train predictions (for residuals)
    y_train_pred = linear_regression_forecast(X_train, y_train, X_train)
    residuals = get_train_residuals(y_train, y_train_pred)

    # Test predictions
    y_test_pred = linear_regression_forecast(X_train, y_train, X_test)

    intervals = calculate_prediction_intervals(y_test_pred, residuals)

    print(intervals.head())
