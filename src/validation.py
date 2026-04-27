import numpy as np

def rolling_validation(model_func, X, y, window=50000, step=5000):
    errors = []

    for start in range(0, len(X) - window - step, step):
        end = start + window

        X_train = X.iloc[start:end]
        y_train = y.iloc[start:end]

        X_test = X.iloc[end:end+step]
        y_test = y.iloc[end:end+step]

        y_pred = model_func(X_train, y_train, X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        errors.append(mae)

    return np.mean(errors)
