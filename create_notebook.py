from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = [
    nbf.v4.new_markdown_cell("# House Price Prediction\n\nIBM SkillsBuild Data Analytics with AI Academic Internship Project. This notebook builds an end-to-end regression solution using `Housing.csv`."),
    nbf.v4.new_markdown_cell("## 1. Import libraries and load the dataset"),
    nbf.v4.new_code_cell("""from pathlib import Path
import json, joblib
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

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "Housing.csv"
if not DATA_PATH.exists(): DATA_PATH = ROOT / "archive (3)" / "Housing.csv"
df = pd.read_csv(DATA_PATH)
df.head()"""),
    nbf.v4.new_code_cell("print(f'Rows: {df.shape[0]}, columns: {df.shape[1]}')\ndisplay(df.describe(include='all').T)\nprint('Missing values:\n', df.isna().sum())"),
    nbf.v4.new_markdown_cell("## 2. Exploratory analysis"),
    nbf.v4.new_code_cell("""sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df["price"], kde=True, ax=axes[0], color="#1f77b4")
axes[0].set_title("Price distribution")
sns.scatterplot(data=df, x="area", y="price", hue="airconditioning", ax=axes[1])
axes[1].set_title("Area and price relationship")
plt.tight_layout()"""),
    nbf.v4.new_markdown_cell("## 3. Prepare features and split the data\n\nNumeric variables are scaled; categorical variables are one-hot encoded. Preprocessing stays inside each model pipeline."),
    nbf.v4.new_code_cell("""X = df.drop(columns=["price"])
y = df["price"]
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

def build_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
    ])"""),
    nbf.v4.new_markdown_cell("## 4. Train and compare models"),
    nbf.v4.new_code_cell("""candidates = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
}
results, fitted = {}, {}
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {"MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2": r2_score(y_test, pred)}
    fitted[name] = pipe
pd.DataFrame(results).T.sort_values("RMSE")"""),
    nbf.v4.new_markdown_cell("## 5. Evaluate the selected model"),
    nbf.v4.new_code_cell("""best_name = min(results, key=lambda name: results[name]["RMSE"])
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
print("Selected model:", best_name)
print(json.dumps({k: float(v) for k, v in results[best_name].items()}, indent=2))
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.65, color="#2ca02c")
low, high = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
plt.plot([low, high], [low, high], "--", color="#d62728")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("Actual vs predicted prices"); plt.show()"""),
    nbf.v4.new_markdown_cell("## 6. Example inference and artifact export"),
    nbf.v4.new_code_cell("""sample_house = pd.DataFrame([{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes", "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}])
print(f"Estimated price: {best_model.predict(sample_house)[0]:,.0f}")
joblib.dump(best_model, ROOT / "model" / "house_price_model.joblib")"""),
    nbf.v4.new_markdown_cell("## Conclusion\n\nThe saved pipeline is used by the Flask API and Streamlit frontend. The output is an educational estimate and should be improved with location, time, neighborhood, and market variables before production use."),
]
nb.cells = cells
nbf.write(nb, Path("house_price_prediction.ipynb"))
print("Created house_price_prediction.ipynb")
