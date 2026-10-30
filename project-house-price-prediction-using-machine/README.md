# House Price Prediction Using Machine Learning

A beginner-friendly regression project that explores housing records, trains a Linear Regression model, evaluates it on held-out data, and serves estimates through Streamlit. Project period: January 2026–February 2026. The dataset does not document a currency or unit; the application labels the target simply as **Price**.

## Dataset

The supplied CSV contains 2,000 records and columns `Id`, `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Location`, `Condition`, `Garage`, and `Price`. `Id` is excluded from modeling. Numeric inputs are Area, Bedrooms, Bathrooms, Floors, and YearBuilt; categorical inputs are Location, Condition, and Garage. `Price` is the target. The loader validates required fields and numeric values, rejects invalid records with a clear error instead of silently dropping them, and leaves the source file untouched.

Initial inspection found 2,000 rows, all required columns, no blank cells or duplicate records. Observed ranges: Area 501–4,999; Bedrooms 1–5; Bathrooms 1–4; Floors 1–3; YearBuilt 1900–2023; Price 50,005–999,656. Location values: Downtown, Rural, Suburban, Urban; Condition: Excellent, Fair, Good, Poor; Garage: No, Yes. These are descriptive properties of this file, not claims about the broader housing market.

## Structure

```text
house-price-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/House Price Prediction Dataset.csv
├── models/                   # generated, ignored by Git
├── outputs/                  # generated metrics, predictions, plots
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── evaluation.py
    └── prediction.py
```

## Setup and execution (Windows)

From this folder in PowerShell or the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Training must complete successfully before starting the application the first time. Keep the original supplied CSV in `data/House Price Prediction Dataset.csv`. Training writes `models/house_price_model.joblib`, model metadata, `outputs/evaluation_metrics.json`, test `outputs/predictions.csv`, feature coefficients, and charts under `outputs/visualizations/`. Generated data and model artifacts are ignored by Git.

## Workflow

The training script prints dimensions, columns, data types, missingness, duplicates, and a preview. It validates values and uses a reproducible 80:20 train/test split (`random_state=42`). A Scikit-learn `Pipeline` holds a `ColumnTransformer`: numeric fields pass through and categorical fields use `OneHotEncoder(handle_unknown="ignore")`. The pipeline and estimator are fit using training data only. The fitted full pipeline is saved with Joblib and reused by Streamlit without retraining.

The script produces EDA charts for Price and Area distributions, numeric correlations, Price versus Area, average Price by Location and Condition, and Price by Garage. It also saves actual-versus-predicted, residual, residual-distribution, and coefficient plots. These are observed associations; coefficient values are conditional on the included features and are not causal effects.

## Evaluation

The held-out test set reports Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared. MAE is average absolute error in the target's original Price units. RMSE is in those same units and penalizes large errors more strongly. MSE uses squared target units. R-squared compares model predictions with predicting the test-set mean; it is not classification accuracy and may be negative. A baseline predicting the training-set mean is evaluated on the same test rows. Actual metrics are generated at training time and stored in `outputs/evaluation_metrics.json`; they are not hardcoded here.

## Example prediction workflow

Open the Streamlit app, select or enter Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, and Garage, then choose **Predict House Price**. The app validates input ranges and uses the saved fitted pipeline. The resulting value is a historical-data model estimate, not a guarantee of real-world market value.

## Limitations and future work

Price units and collection date are unspecified. The available features may omit major market drivers, and synthetic or sampled patterns may not generalize to other regions or periods. Linear Regression assumes a linear relationship in its encoded feature space. Future work could investigate stronger data documentation, richer features, cross-validation, regularized or nonlinear models, and monitoring with representative fresh records. The project should not be described as production-ready.
