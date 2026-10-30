"""Streamlit dashboard for exploring the dataset and estimating house prices."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_dataset
from src.prediction import predict_one

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "House Price Prediction Dataset.csv"
MODEL = ROOT / "models" / "house_price_model.joblib"
METRICS = ROOT / "outputs" / "evaluation_metrics.json"
PLOTS = ROOT / "outputs" / "visualizations"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("House Price Prediction Using Machine Learning")
st.caption("Historical data estimates only; results are not a guarantee of real-world market value.")
if not DATA.is_file(): st.error("Dataset is missing. Add it at data/House Price Prediction Dataset.csv."); st.stop()
try: dataset = load_dataset(DATA)
except Exception as exc: st.error(str(exc)); st.stop()
sections = st.tabs(["Overview", "Dataset Explorer", "Model Performance", "Predict", "Documentation"])
with sections[0]:
    st.subheader("Project overview")
    st.write("Estimate recorded house prices from property characteristics with a supervised Linear Regression model.")
    a,b,c=st.columns(3); a.metric("Dataset records", f"{len(dataset):,}"); b.metric("Input features", len(FEATURES)); c.metric("Target", "Price")
    st.write("**Algorithm:** Linear Regression  |  **Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit")
    st.write("**Features:** Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage. Id is an identifier and is excluded.")
with sections[1]:
    st.subheader("Dataset explorer")
    st.write(f"Dimensions: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    st.dataframe(dataset.head(100), use_container_width=True)
    st.subheader("Numerical summary"); st.dataframe(dataset[NUMERIC_FEATURES + ["Price"]].describe(), use_container_width=True)
    st.subheader("Missing values"); st.dataframe(dataset.isna().sum().rename("Missing values"))
    for field in CATEGORICAL_FEATURES: st.write(f"**{field}:** " + ", ".join(map(str, sorted(dataset[field].unique()))))
with sections[2]:
    st.subheader("Held-out test performance")
    if not METRICS.is_file(): st.info("Train the model first: `python train_model.py`.")
    else:
        metrics=json.loads(METRICS.read_text(encoding="utf-8"))
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[("MAE","mae"),("MSE","mse"),("RMSE","rmse"),("R-squared","r2")]): col.metric(label, f"{metrics[key]:,.3f}")
        st.caption("MAE and RMSE are in the target's original Price units; MSE is in squared Price units. R-squared compares with predicting the test-set mean; negative values are possible and indicate performance worse than that mean baseline.")
        base=metrics["baseline_mean_price"]
        st.write(f"Mean-price baseline — MAE {base['mae']:,.3f}, RMSE {base['rmse']:,.3f}. Linear Regression MAE improvement: {metrics['comparison']['mae_improvement']:,.3f}; RMSE improvement: {metrics['comparison']['rmse_improvement']:,.3f}.")
        st.write("Coefficients describe conditional model associations and should not be interpreted as causal effects.")
        for filename,caption in [("actual_vs_predicted.png","Actual versus predicted"),("residual_plot.png","Residuals versus predicted"),("residual_distribution.png","Residual distribution")]:
            path=PLOTS/filename
            if path.is_file(): st.image(str(path), caption=caption, use_container_width=True)
with sections[3]:
    st.subheader("Estimate a house price")
    if not MODEL.is_file(): st.info("The fitted model is not available. Run `python train_model.py` from the project directory, then restart this page.")
    else:
        categories={c: sorted(dataset[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
        with st.form("prediction_form"):
            c1,c2=st.columns(2)
            area=c1.number_input("Area", min_value=0.01, value=1500.0, step=100.0)
            bedrooms=c1.number_input("Bedrooms", min_value=0, max_value=50, value=3, step=1)
            bathrooms=c1.number_input("Bathrooms", min_value=0, max_value=50, value=2, step=1)
            floors=c1.number_input("Floors", min_value=0, max_value=50, value=1, step=1)
            year=c2.number_input("YearBuilt", min_value=1800, max_value=2026, value=2000, step=1)
            location=c2.selectbox("Location", categories["Location"])
            condition=c2.selectbox("Condition", categories["Condition"])
            garage=c2.selectbox("Garage", categories["Garage"])
            submitted=st.form_submit_button("Predict House Price")
        if submitted:
            try:
                fitted=joblib.load(MODEL)
                estimate=predict_one(fitted,{"Area":area,"Bedrooms":bedrooms,"Bathrooms":bathrooms,"Floors":floors,"YearBuilt":year,"Location":location,"Condition":condition,"Garage":garage})
                st.success(f"Estimated Price: {estimate:,.2f}")
                st.caption("This is a model estimate based on historical records. Actual market prices may differ.")
            except Exception as exc: st.error(f"Prediction failed: {exc}")
with sections[4]:
    st.subheader("How the project works")
    st.markdown("""- **Supervised learning:** the model learns a mapping from recorded house features to historical Price values.
- **Linear Regression:** fits coefficients and an intercept to minimize squared errors on training data.
- **Preprocessing:** numeric columns pass through; Location, Condition, and Garage use one-hot encoding. Unknown categories are safely ignored by the fitted encoder.
- **Training and testing:** an 80:20 split with random_state=42. Preprocessing is fitted as part of a pipeline on training rows only; test data is reserved for evaluation.
- **Metrics:** MAE is average absolute error in Price units; MSE averages squared errors; RMSE is the square root of MSE in Price units; R-squared compares with a mean-prediction baseline and may be negative.
- **Limitations:** relationships are associations, historical data may not reflect current markets, and important factors may be absent. Price units were not documented. The model is not a valuation guarantee.""")
