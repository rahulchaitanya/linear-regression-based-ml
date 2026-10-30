"""Regression metrics and diagnostics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, predicted))}


def mean_baseline_metrics(y_train, y_test):
    prediction = np.full(len(y_test), float(np.mean(y_train)))
    return regression_metrics(y_test, prediction), prediction
