# House Price Prediction

> An end-to-end machine-learning project that predicts residential house prices from property features using Python, scikit-learn, Flask, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Backend](https://img.shields.io/badge/API-Flask-000000?logo=flask&logoColor=white)
![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Overview

This project uses the supplied `Housing.csv` dataset to estimate the price of a house from its area, room count, facilities, road access, parking, preferred-area status, and furnishing status.

The project includes:

- A complete Jupyter Notebook for academic submission.
- A reproducible training and evaluation script.
- A saved scikit-learn preprocessing-and-model pipeline.
- A Flask REST API for predictions.
- A Streamlit interface for interactive predictions.
- Evaluation charts and a formatted Word project report.

> **Project status:** Complete and tested locally. The model and API are ready to run after installing the dependencies.

## Project structure

```text
HOUSE_PREDICTION/
|
|-- data/
|   `-- Housing.csv                     # Source dataset
|
|-- model/
|   |-- house_price_model.joblib        # Saved trained pipeline
|   `-- metrics.json                    # Metrics and model metadata
|
|-- outputs/
|   |-- actual_vs_predicted.png         # Evaluation chart
|   `-- price_distribution.png          # Price distribution chart
|
|-- app.py                              # Flask REST API
|-- ui.py                               # Streamlit frontend
|-- train_model.py                      # Training and evaluation script
|-- house_price_prediction.ipynb        # Complete notebook submission
|-- requirements.txt                    # Python dependencies
|-- README.md                           # Project documentation
|-- House_Price_Prediction_Report.docx  # Final report
|-- PROJECT_REPORT.docx                 # Report copy
|-- create_notebook.py                  # Notebook generator helper
`-- build_report.py                     # Report generator helper
```

## Dataset

The source dataset is available at [`data/Housing.csv`](data/Housing.csv).

| Property | Value |
|---|---:|
| Records | 545 |
| Columns | 13 |
| Input features | 12 |
| Target column | `price` |
| Missing values | 0 |
| Average price | 4,766,729 |

### Input features

| Type | Features |
|---|---|
| Numeric | `area`, `bedrooms`, `bathrooms`, `stories`, `parking` |
| Binary categorical | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` |
| Multi-class categorical | `furnishingstatus` |

## How the project works

```text
Housing.csv
    |
    v
Data loading and validation
    |
    v
Preprocessing: imputation, scaling, one-hot encoding
    |
    v
Model comparison: Linear Regression vs Random Forest
    |
    v
Best model saved with joblib
    |
    +--> Flask API --> JSON prediction
    |
    `--> Streamlit UI --> User-friendly prediction form
```

1. Load `data/Housing.csv` into pandas.
2. Separate the target column, `price`, from the input features.
3. Split the data into 80% training and 20% testing sets using `random_state=42`.
4. Impute and scale numeric features.
5. Impute and one-hot encode categorical features.
6. Compare Linear Regression and Random Forest models.
7. Select the model with the lower test RMSE.
8. Save the complete pipeline to `model/house_price_model.joblib`.
9. Serve predictions through Flask and display them through Streamlit.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project

### 1. Train the model

```bash
python train_model.py
```

This generates:

- `model/house_price_model.joblib`
- `model/metrics.json`
- `outputs/actual_vs_predicted.png`
- `outputs/price_distribution.png`

### 2. Start the Flask API

```bash
python app.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

### 3. Start the Streamlit frontend

Open a second terminal while the API is running:

```bash
streamlit run ui.py
```

The frontend opens at:

```text
http://localhost:8501
```

### 4. Open the notebook

```bash
jupyter notebook house_price_prediction.ipynb
```

## API documentation

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the API and model are ready |
| `GET` | `/dataset` | Return the training dataset as JSON |
| `GET` | `/model-info` | Return selected model and evaluation metrics |
| `GET` | `/model_info` | Compatibility alias for `/model-info` |
| `POST` | `/predict` | Return a predicted house price |

### Health check

```bash
curl http://127.0.0.1:5000/health
```

Example response:

```json
{
  "model_ready": true,
  "status": "ok"
}
```

### Prediction request

```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}
```

Example response:

```json
{
  "predicted_price": 7968276.13,
  "currency": "dataset currency units"
}
```

## Frontend Pages

| Page | Description |
|---|---|
| **Predict Price** | Enter all property details and receive an instant estimate from the Flask API. |
| **Input Features** | Area, bedrooms, bathrooms, stories, parking, facilities, road access, and furnishing status. |
| **Prediction Result** | Display the estimated price returned by the `/predict` endpoint. |
| **Future Dashboard** | Dataset exploration and model-insights pages can be added as the next UI enhancement. |

The frontend sends the form data to the Flask `/predict` endpoint and displays the returned estimate.

## Model Performance

The models are evaluated on the held-out test set using MAE, RMSE, and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 970,043 | 1,324,507 | 0.653 |
| Random Forest | 1,044,103 | 1,407,532 | 0.608 |

### Selected model

**Linear Regression** was selected because it achieved the lower test RMSE for the fixed dataset split.

> The metrics are estimates for this dataset and split. They should not be interpreted as a guarantee of performance on new markets or locations.

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data processing | pandas, NumPy | Load and prepare tabular data |
| Machine learning | scikit-learn | Preprocessing, training, and evaluation |
| Model persistence | joblib | Save and load the trained pipeline |
| Backend | Flask | Serve REST API endpoints |
| Frontend | Streamlit | Provide the interactive UI |
| Visualisation | Matplotlib, Seaborn | Generate charts and diagnostics |
| Documentation | Jupyter, python-docx | Notebook and project report |

## Submission checklist

- [x] `house_price_prediction.ipynb`
- [x] `requirements.txt`
- [x] `README.md`
- [x] `House_Price_Prediction_Report.docx`
- [x] `data/Housing.csv`
- [x] `train_model.py`
- [x] `app.py`
- [x] `ui.py`
- [x] Trained model and evaluation outputs

## Limitations and future improvements

The dataset does not include exact location, neighbourhood, construction year, market date, economic conditions, or detailed property condition. Predictions are educational estimates and should not be used as professional valuations, lending decisions, or guarantees of market price.

Potential improvements include:

- Add location and neighbourhood features.
- Use cross-validation and hyperparameter tuning.
- Add prediction intervals and uncertainty reporting.
- Add data-drift and model-performance monitoring.
- Expand the frontend with dataset exploration and model-insights pages.

