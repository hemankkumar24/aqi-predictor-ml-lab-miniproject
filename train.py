# ==============================================================================
# AirCast India - Model Training Script (train.py)
#
# This script trains 4 Machine Learning models to predict tomorrow's AQI
# across Indian cities using historical air quality data (2015-2020).
#
# It performs:
#   1. Automatic dataset download from Kaggle (if not already present in data/)
#   2. Data cleaning & robust median imputation
#   3. Target variable creation (tomorrow's AQI via shift within each city)
#   4. Time-based train-test split (80% train, 20% test)
#   5. Training 4 regression models (Linear Reg, Decision Tree, Random Forest, GBDT)
#   6. Evaluation against a Naive Baseline ("tomorrow = today")
#   7. Exporting trained models (models.pkl) and performance metrics (metrics.json)
# ==============================================================================

import os
import shutil
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# ------------------------------------------------------------------------------
# 1. DOWNLOAD DATASET IF MISSING
# ------------------------------------------------------------------------------
# We check if data/city_day.csv exists. If not, we attempt to download it
# automatically from Kaggle using the kagglehub package.
# ------------------------------------------------------------------------------
def ensure_dataset():
    data_dir = "data"
    target_csv = os.path.join(data_dir, "city_day.csv")

    if os.path.exists(target_csv):
        print(f"[✓] Dataset already exists at: {target_csv}")
        return target_csv

    print("[i] 'data/city_day.csv' not found. Attempting download via kagglehub...")
    os.makedirs(data_dir, exist_ok=True)

    try:
        import kagglehub
        # Download the specific dataset slug
        download_path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
        src_file = os.path.join(download_path, "city_day.csv")

        if os.path.exists(src_file):
            shutil.copy(src_file, target_csv)
            print(f"[✓] Successfully downloaded and copied city_day.csv to {target_csv}")
            return target_csv
        else:
            raise FileNotFoundError(f"city_day.csv was not found inside downloaded folder: {download_path}")

    except Exception as err:
        print("\n" + "=" * 70)
        print("ERROR: Automatic Kaggle download failed!")
        print(f"Details: {err}")
        print("\nPlease either:")
        print("  1. Configure your Kaggle credentials (~/.kaggle/kaggle.json)")
        print("  2. OR manually download 'city_day.csv' from:")
        print("     https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
        print("     and place it directly into the 'data/' directory.")
        print("=" * 70 + "\n")
        raise SystemExit(1)


# ------------------------------------------------------------------------------
# 2. LOAD AND SORT HISTORICAL DATA
# ------------------------------------------------------------------------------
# Load the daily city-level air quality observations, parse dates properly,
# and sort chronologically within each city to allow time-series shifting.
# ------------------------------------------------------------------------------
def load_and_preprocess_data(csv_path):
    print("\n--- Step 2: Loading dataset ---")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} raw records across {df['City'].nunique()} cities.")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically within each city
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # 3. CLEANING & IMPUTATION
    # --------------------------------------------------------------------------
    # - Drop rows where the target AQI is missing (we cannot train on empty ground truth).
    # - Impute missing pollutant values with each city's median (robust to extreme spikes).
    # - If a city is missing an entire pollutant column, fallback to the national median.
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Cleaning & Missing Value Imputation ---")
    initial_rows = len(df)
    df = df.dropna(subset=["AQI"]).copy()
    print(f"Dropped {initial_rows - len(df):,} rows with missing AQI. Remaining: {len(df):,}")

    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

    for pollutant in pollutant_cols:
        # Group-wise median for the specific city
        city_median = df.groupby("City")[pollutant].transform("median")
        df[pollutant] = df[pollutant].fillna(city_median)

        # Global median fallback in case the entire city had no readings for this pollutant
        global_median = df[pollutant].median()
        df[pollutant] = df[pollutant].fillna(global_median)

    # --------------------------------------------------------------------------
    # 4. TARGET VARIABLE: AQI OF TOMORROW
    # --------------------------------------------------------------------------
    # We shift the AQI column backwards by 1 within each city.
    # That means for today's row, 'AQI_tomorrow' is the actual AQI measured the next day.
    # We drop the very last day of each city because tomorrow has not happened yet!
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Creating Target Variable (AQI_tomorrow) ---")
    df["AQI_tomorrow"] = df.groupby("City")["AQI"].shift(-1)
    df = df.dropna(subset=["AQI_tomorrow"]).copy()
    print(f"Dataset size with valid tomorrow target: {len(df):,} rows.")

    # --------------------------------------------------------------------------
    # 5. FEATURE SELECTION
    # --------------------------------------------------------------------------
    # We select 9 core pollutant features plus 'Month' (to capture seasonal smog/monsoon).
    #
    # WHY SKIP BENZENE, TOLUENE, XYLENE?
    # Benzene, Toluene, and Xylene (BTX) are missing in >60% of Indian monitoring stations
    # and are not standard outputs of global atmospheric models like Open-Meteo.
    #
    # WHY NOT USE TODAY'S AQI AS A FEATURE?
    # If we include today's AQI, the model would simply predict "tomorrow ≈ today" and
    # ignore individual chemical pollutants. Excluding today's AQI forces the model
    # to learn the genuine physical relationship between chemical concentrations and AQI.
    # Also, we do NOT use city name so our model generalizes to any Indian coordinate!
    # --------------------------------------------------------------------------
    print("\n--- Step 5: Preparing Feature Columns ---")
    df["Month"] = df["Date"].dt.month
    feature_names = pollutant_cols + ["Month"]
    print(f"Features used ({len(feature_names)}): {', '.join(feature_names)}")

    return df, feature_names


# ------------------------------------------------------------------------------
# 6. TIME-BASED TRAIN / TEST SPLIT
# ------------------------------------------------------------------------------
# We do NOT use a random train_test_split.
# Why? Because in time-series forecasting, "we can't use the future to predict the past".
# A random split causes data leakage where the model cheats using future weather patterns.
# Instead, we sort strictly by Date: the first 80% is train, the last 20% is test.
# ------------------------------------------------------------------------------
def split_data(df, feature_names):
    print("\n--- Step 6: Time-based Train / Test Split ---")
    df = df.sort_values(by="Date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_names]
    y_train = train_df["AQI_tomorrow"]

    X_test = test_df[feature_names]
    y_test = test_df["AQI_tomorrow"]

    print(f"Training set:   {len(train_df):,} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Test set:       {len(test_df):,} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    return X_train, y_train, X_test, y_test, test_df


# ------------------------------------------------------------------------------
# 7 & 8. TRAIN AND EVALUATE 4 MODELS + NAIVE BASELINE
# ------------------------------------------------------------------------------
# We compare:
#   1. Linear Regression (baseline parametric model with StandardScaler)
#   2. Decision Tree (non-linear baseline with max_depth=10)
#   3. Random Forest (ensemble bagging of 200 trees, handles non-linear interactions)
#   4. Gradient Boosting (ensemble boosting, iteratively reduces residuals)
# ------------------------------------------------------------------------------
def train_and_evaluate_models(X_train, y_train, X_test, y_test, test_df, feature_names):
    print("\n--- Step 7: Training Models ---")

    # Define the 4 model pipelines / estimators
    model_dict = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    metrics = {}
    feature_importances = {}
    test_eval_df = test_df[["City", "Date", "AQI", "AQI_tomorrow"]].copy()

    # Naive Baseline: assume tomorrow's AQI is identical to today's AQI
    naive_predictions = test_df["AQI"]
    naive_mae = float(mean_absolute_error(y_test, naive_predictions))
    metrics["naive_baseline_mae"] = round(naive_mae, 2)

    print("\n" + "=" * 68)
    print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
    print("=" * 68)
    print(f"{'Naive (Today=Tomorrow)':<22} | {naive_mae:<8.2f} | {'N/A':<8} | {'N/A':<8}")
    print("-" * 68)

    for name, model in model_dict.items():
        # Fit model on training data
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        test_eval_df[f"Pred_{name}"] = y_pred

        # Calculate metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics[name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        }

        print(f"{name:<22} | {mae:<8.2f} | {rmse:<8.2f} | {r2:<8.4f}")

        # Extract feature importances
        # For tree models: feature_importances_
        # For linear regression: absolute standardized coefficients
        if name == "Linear Regression":
            regressor = model.named_steps["regressor"]
            importances = np.abs(regressor.coef_)
        else:
            importances = model.feature_importances_

        # Normalize to sum to 1.0 or store as dictionary mapping feature to weight
        total_imp = importances.sum()
        norm_imp = (importances / total_imp) if total_imp > 0 else importances
        feature_importances[name] = {feat: round(float(val), 4) for feat, val in zip(feature_names, norm_imp)}

    print("=" * 68 + "\n")

    return model_dict, metrics, feature_importances, test_eval_df


# ------------------------------------------------------------------------------
# 9. SAVE ARTIFACTS (metrics.json & models.pkl)
# ------------------------------------------------------------------------------
# Save all trained models, slider statistics for the app sidebar, feature names,
# and metrics comparison.
# ------------------------------------------------------------------------------
def save_artifacts(model_dict, metrics, feature_importances, full_df, feature_names, test_eval_df):
    print("--- Step 9: Saving artifacts ---")

    # 1. Slider statistics for features not provided by the live API (NO, NOx, NH3)
    # Range = dataset min to 95th percentile; default = dataset median
    slider_info = {}
    for col in ["NO", "NOx", "NH3"]:
        slider_info[col] = {
            "min": round(float(full_df[col].min()), 2),
            "max": round(float(full_df[col].quantile(0.95)), 2),
            "median": round(float(full_df[col].median()), 2)
        }

    # 2. Save metrics.json
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[✓] Saved metrics to: {metrics_path}")

    # 3. Save models.pkl with joblib
    models_path = "models.pkl"
    payload = {
        "models": model_dict,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "slider_info": slider_info,
        "naive_baseline_mae": metrics["naive_baseline_mae"],
        "test_eval_df": test_eval_df  # Used for Tab 3 actual vs predicted test chart
    }
    joblib.dump(payload, models_path)
    print(f"[✓] Saved trained models & metadata to: {models_path}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION FLOW
# ------------------------------------------------------------------------------
def main():
    print("==================================================================")
    print("        AirCast India - AQI Tomorrow Prediction Training         ")
    print("==================================================================")

    # 1. Ensure dataset exists
    csv_path = ensure_dataset()

    # 2-5. Load, clean, engineer target and features
    df, feature_names = load_and_preprocess_data(csv_path)

    # 6. Time-based split
    X_train, y_train, X_test, y_test, test_df = split_data(df, feature_names)

    # 7-8. Train all 4 models and evaluate
    model_dict, metrics, feature_importances, test_eval_df = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, test_df, feature_names
    )

    # 9. Save metrics.json and models.pkl
    save_artifacts(model_dict, metrics, feature_importances, df, feature_names, test_eval_df)

    print("\n[✓] All training steps completed successfully!")


if __name__ == "__main__":
    main()
