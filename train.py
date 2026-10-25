import os
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


def load_and_preprocess_data(csv_path="data/city_day.csv"):
    # read the dataset containing daily air quality records across Indian cities
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} raw records across {df['City'].nunique()} cities.")

    # convert Date column to datetime so pandas can handle time-series operations
    df["Date"] = pd.to_datetime(df["Date"])

    # sort chronologically by city and date so time-based splitting works properly
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)

    # remove rows where the target AQI is missing because we can't train without ground truth
    initial_rows = len(df)
    df = df.dropna(subset=["AQI"]).copy()
    print(f"Dropped {initial_rows - len(df):,} rows with missing AQI. Remaining: {len(df):,}")

    # calculate yesterday's AQI within each city for the persistence benchmark (naive baseline)
    df["AQI_yesterday"] = df.groupby("City")["AQI"].shift(1)

    # list of 9 core pollutants we are using as features
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

    # missing value imputation:
    # 1. replace missing pollutant values with the city's median (median is robust against extreme pollution spikes)
    # 2. if a city has no data at all for a pollutant, fallback to the overall median across all cities
    for pollutant in pollutant_cols:
        city_median = df.groupby("City")[pollutant].transform("median")
        df[pollutant] = df[pollutant].fillna(city_median)

        global_median = df[pollutant].median()
        df[pollutant] = df[pollutant].fillna(global_median)

    # extract month from date to help the model learn seasonal patterns (e.g., winter smog vs monsoon)
    df["Month"] = df["Date"].dt.month
    feature_names = pollutant_cols + ["Month"]
    print(f"Features used ({len(feature_names)}): {', '.join(feature_names)}")

    return df, feature_names


def split_data(df, feature_names):
    # time-based split: sort all records strictly by date
    # we don't use random train_test_split here to prevent data leakage (we shouldn't use future data to predict the past)
    df = df.sort_values(by="Date").reset_index(drop=True)

    # first 80% of dates for training, last 20% for testing
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # separate input features (X) and target variable (y) - predicting current instant's AQI
    X_train = train_df[feature_names]
    y_train = train_df["AQI"]

    X_test = test_df[feature_names]
    y_test = test_df["AQI"]

    print(f"Training set:   {len(train_df):,} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Test set:       {len(test_df):,} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    return X_train, y_train, X_test, y_test, test_df


def train_and_evaluate_models(X_train, y_train, X_test, y_test, test_df, feature_names):
    # setting up 4 different regression models to compare their performance:
    # 1. Linear Regression: baseline linear model; we wrap it in a Pipeline with StandardScaler because scale matters for regression
    # 2. Decision Tree: non-linear model; max_depth=10 prevents the tree from overfitting too deeply
    # 3. Random Forest: ensemble bagging method; combines 200 trees to reduce variance and handle complex interactions
    # 4. Gradient Boosting: ensemble boosting method; builds trees sequentially, each correcting the previous tree's errors
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
    test_eval_df = test_df[["City", "Date", "AQI"]].copy()
    test_eval_df["AQI_tomorrow"] = test_eval_df["AQI"]  # backwards compatibility

    # naive baseline: persistence benchmark (predicting current AQI using yesterday's AQI)
    # missing yesterday values fallback to training set median
    naive_predictions = test_df["AQI_yesterday"].fillna(y_train.median())
    naive_mae = float(mean_absolute_error(y_test, naive_predictions))
    metrics["naive_baseline_mae"] = round(naive_mae, 2)

    print("\n" + "=" * 68)
    print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
    print("=" * 68)
    print(f"{'Naive (Yesterday AQI)':<22} | {naive_mae:<8.2f} | {'N/A':<8} | {'N/A':<8}")
    print("-" * 68)

    # train each model and evaluate on the unseen test set
    for name, model in model_dict.items():
        # fit the model on our training data
        model.fit(X_train, y_train)

        # generate predictions on test data
        y_pred = model.predict(X_test)
        test_eval_df[f"Pred_{name}"] = y_pred

        # evaluation metrics:
        # MAE: Mean Absolute Error (average magnitude of errors in AQI points)
        # RMSE: Root Mean Squared Error (penalizes large outlier errors more heavily)
        # R2: Coefficient of Determination (proportion of variance explained by the model)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics[name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        }

        print(f"{name:<22} | {mae:<8.2f} | {rmse:<8.2f} | {r2:<8.4f}")

        # calculate feature importances:
        # for tree models we use feature_importances_
        # for linear regression we take the absolute value of the standardized coefficients
        if name == "Linear Regression":
            regressor = model.named_steps["regressor"]
            importances = np.abs(regressor.coef_)
        else:
            importances = model.feature_importances_

        # normalize importances so they sum to 1.0 (easy to compare as percentages)
        total_imp = importances.sum()
        norm_imp = (importances / total_imp) if total_imp > 0 else importances
        feature_importances[name] = {feat: round(float(val), 4) for feat, val in zip(feature_names, norm_imp)}

    print("=" * 68 + "\n")

    return model_dict, metrics, feature_importances, test_eval_df


def save_artifacts(model_dict, metrics, feature_importances, full_df, feature_names, test_eval_df):
    # compute min, max, and median for pollutants not returned by live APIs (NO, NOx, NH3)
    # these are used in the Streamlit app sidebar sliders so the user has realistic default values
    slider_info = {}
    for col in ["NO", "NOx", "NH3"]:
        slider_info[col] = {
            "min": round(float(full_df[col].min()), 2),
            "max": round(float(full_df[col].quantile(0.95)), 2),
            "median": round(float(full_df[col].median()), 2)
        }

    # save evaluation metrics to a json file for easy display in the web app
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[✓] Saved metrics to: {metrics_path}")

    # save trained models, feature names, importances, and test predictions into models.pkl using joblib
    models_path = "models.pkl"
    payload = {
        "models": model_dict,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "slider_info": slider_info,
        "naive_baseline_mae": metrics["naive_baseline_mae"],
        "test_eval_df": test_eval_df
    }
    joblib.dump(payload, models_path)
    print(f"[✓] Saved trained models & metadata to: {models_path}")


def main():
    print("Starting AQI prediction model training...")

    # path to our dataset
    csv_path = "data/city_day.csv"

    # load and preprocess dataset
    df, feature_names = load_and_preprocess_data(csv_path)

    # split into train and test sets chronologically
    X_train, y_train, X_test, y_test, test_df = split_data(df, feature_names)

    # train and evaluate our 4 regression models
    model_dict, metrics, feature_importances, test_eval_df = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, test_df, feature_names
    )

    # save the models and evaluation metrics for deployment
    save_artifacts(model_dict, metrics, feature_importances, df, feature_names, test_eval_df)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
