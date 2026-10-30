"""Train, evaluate, and save the house price regression pipeline."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.evaluation import mean_baseline_metrics, regression_metrics
from src.preprocessing import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                               TARGET, load_dataset, make_preprocessor)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "House Price Prediction Dataset.csv"
OUT = ROOT / "outputs"
PLOTS = OUT / "visualizations"
MODEL = ROOT / "models" / "house_price_model.joblib"


def save_eda(data):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(data=data, x=TARGET, ax=ax); ax.set(title="Distribution of Recorded House Prices", xlabel="Price", ylabel="Number of houses"); fig.tight_layout(); fig.savefig(PLOTS / "price_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(data=data, x="Area", ax=ax); ax.set(title="Distribution of House Area", xlabel="Area", ylabel="Number of houses"); fig.tight_layout(); fig.savefig(PLOTS / "area_distribution.png", dpi=150); plt.close(fig)
    numeric = data[NUMERIC_FEATURES + [TARGET]]
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax); ax.set_title("Correlation Among Numeric Features and Price"); fig.tight_layout(); fig.savefig(PLOTS / "correlation_heatmap.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=data, x="Area", y=TARGET, alpha=.65, ax=ax); ax.set(title="Recorded Price versus Area", xlabel="Area", ylabel="Price"); fig.tight_layout(); fig.savefig(PLOTS / "price_vs_area.png", dpi=150); plt.close(fig)
    for field, filename, title in [("Location", "average_price_by_location.png", "Average Recorded Price by Location"), ("Condition", "average_price_by_condition.png", "Average Recorded Price by Condition")]:
        grouped = data.groupby(field, observed=True)[TARGET].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5)); grouped.plot(kind="bar", ax=ax); ax.set(title=title, xlabel=field, ylabel="Average Price"); fig.tight_layout(); fig.savefig(PLOTS / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=data, x="Garage", y=TARGET, ax=ax); ax.set(title="Recorded Price Distribution by Garage Availability", xlabel="Garage", ylabel="Price"); fig.tight_layout(); fig.savefig(PLOTS / "price_by_garage.png", dpi=150); plt.close(fig)


def main():
    for directory in (OUT, PLOTS, MODEL.parent): directory.mkdir(parents=True, exist_ok=True)
    data = load_dataset(DATA)
    print(f"Shape: {data.shape}\nColumns: {list(data.columns)}\nDtypes:\n{data.dtypes}\n\nFirst five rows:\n{data.head().to_string(index=False)}")
    print(f"\nMissing cells: {int(data.isna().sum().sum())}; duplicate rows: {int(data.duplicated().sum())}")
    # Preserve original data. No bad rows are silently discarded; validation raises with an explanation.
    x, y = data[FEATURES].copy(), data[TARGET].copy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    model = Pipeline([("preprocessing", make_preprocessor()), ("regression", LinearRegression())])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = regression_metrics(y_test, predicted)
    baseline, baseline_prediction = mean_baseline_metrics(y_train, y_test)
    metrics.update({"baseline_mean_price": baseline, "baseline_train_mean": float(y_train.mean()),
                    "comparison": {"mae_improvement": float(baseline["mae"] - metrics["mae"]),
                                   "rmse_improvement": float(baseline["rmse"] - metrics["rmse"]),
                                   "better_mae_than_baseline": metrics["mae"] < baseline["mae"],
                                   "better_rmse_than_baseline": metrics["rmse"] < baseline["rmse"]},
                    "dataset_rows": int(len(data)), "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
                    "features": FEATURES, "target": TARGET, "algorithm": "LinearRegression",
                    "split": {"test_size": .2, "random_state": 42}, "price_unit": "unknown; reported as Price"})
    OUT.joinpath("evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"Actual Price": y_test.to_numpy(), "Predicted Price": predicted,
                  "Signed Prediction Error": predicted - y_test.to_numpy(),
                  "Absolute Prediction Error": np.abs(predicted - y_test.to_numpy())}).to_csv(OUT / "predictions.csv", index=False)
    pre = model.named_steps["preprocessing"]
    feature_names = pre.get_feature_names_out()
    coef = model.named_steps["regression"].coef_
    coefficient_data = pd.DataFrame({"Feature": feature_names, "Coefficient": coef}).sort_values("Coefficient")
    coefficient_data.to_csv(OUT / "feature_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, max(5, len(coef) * .32))); ax.barh(coefficient_data["Feature"], coefficient_data["Coefficient"]); ax.axvline(0, color="black", linewidth=.8); ax.set(title="Linear Regression Feature Coefficients", xlabel="Coefficient (Price units per feature unit)", ylabel="Encoded feature"); fig.tight_layout(); fig.savefig(PLOTS / "feature_coefficients.png", dpi=150); plt.close(fig)
    # Test set diagnostics.
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(y_test, predicted, alpha=.7); lo=min(float(y_test.min()), float(predicted.min())); hi=max(float(y_test.max()), float(predicted.max())); ax.plot([lo,hi],[lo,hi], "r--", label="Perfect prediction"); ax.set(title="Actual versus Predicted Prices", xlabel="Actual Price", ylabel="Predicted Price"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "actual_vs_predicted.png", dpi=150); plt.close(fig)
    residual = y_test.to_numpy() - predicted
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(predicted, residual, alpha=.7); ax.axhline(0, color="red", linestyle="--"); ax.set(title="Residuals versus Predicted Price", xlabel="Predicted Price", ylabel="Residual (Actual - Predicted)"); fig.tight_layout(); fig.savefig(PLOTS / "residual_plot.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); sns.histplot(residual, kde=True, ax=ax); ax.set(title="Distribution of Test Set Residuals", xlabel="Residual (Actual - Predicted)", ylabel="Number of houses"); fig.tight_layout(); fig.savefig(PLOTS / "residual_distribution.png", dpi=150); plt.close(fig)
    save_eda(data)
    joblib.dump(model, MODEL)
    metadata = {"features": FEATURES, "target": TARGET, "algorithm": "LinearRegression", "dataset_rows": len(data), "train_rows": len(x_train), "test_rows": len(x_test), "random_state": 42, "coefficients_csv": "outputs/feature_coefficients.csv", "intercept": float(model.named_steps["regression"].intercept_)}
    MODEL.with_name("model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTest metrics:"); print(json.dumps(metrics, indent=2)); print(f"\nSaved fitted pipeline: {MODEL}")

if __name__ == "__main__": main()
