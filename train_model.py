"""Train, evaluate, and save the house price prediction pipeline."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Housing.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "price"


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(y_test, pipeline.predict(X_test))
        fitted[name] = pipeline

    best_name = min(results, key=lambda name: results[name]["rmse"])
    best_pipeline = fitted[best_name]
    y_pred = best_pipeline.predict(X_test)
    joblib.dump(best_pipeline, MODEL_DIR / "house_price_model.joblib")

    metadata = {
        "target": TARGET,
        "dataset_rows": int(len(df)),
        "dataset_columns": df.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": 0.20,
        "random_state": 42,
        "selected_model": best_name,
        "metrics": results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, color="#1f77b4")
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.65, color="#2ca02c")
    low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([low, high], [low, high], "--", color="#d62728", label="Perfect prediction")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    if best_name == "random_forest":
        preprocessor = best_pipeline.named_steps["preprocessor"]
        model = best_pipeline.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        importance = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(12)
        plt.figure(figsize=(8, 5))
        importance.sort_values().plot.barh(color="#9467bd")
        plt.title("Top Model Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
        plt.close()

    print(json.dumps({"selected_model": best_name, "metrics": results}, indent=2))


if __name__ == "__main__":
    main()
