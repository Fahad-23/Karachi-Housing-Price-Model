# Karachi Housing Price Prediction

ML-powered property price prediction for Karachi real estate. Compares four models — Linear Regression, Random Forest, Gradient Boosting, and a Keras Neural Network — automatically selects the best performer, and serves predictions via a FastAPI REST API and interactive Streamlit dashboard.

[Portfolio Video Demo](https://drive.google.com/file/d/15-3-MQ3LWkqQGWBu4znCwrXS7D_JIZtI/view?usp=sharing)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python train.py
```

Outputs a comparison table of all models with R², RMSE, MAE, and cross-validation scores. Saves the best model to `models/`.

### 2. Start the API

```bash
uvicorn api:app --reload --port 8000
```

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Opens an interactive UI at `http://localhost:8501` with price prediction, model comparison, and data exploration charts.

### 4. Make predictions (CLI)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"property_type": "House", "location": "DHA Defence", "area": 500, "beds": 5, "baths": 4}'
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model type, metrics, known locations/types |
| `POST` | `/predict` | Predict property price |

### Docker

```bash
docker build -t karachi-housing .
docker run -p 8000:8000 karachi-housing
```

Docker is included for reproducible deployment — the container trains the model at build time and starts the API on launch, so anyone can run it without installing Python/TensorFlow locally.

## Models & Hyperparameters

All models are trained on an 80/20 train-test split (`random_state=42`) with cross-validation on the training set.

### 1. Linear Regression (baseline)
- No hyperparameters — standard OLS fit
- 5-fold cross-validation

### 2. Random Forest Regressor
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 15 |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |
- 5-fold cross-validation

### 3. Gradient Boosting Regressor
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `random_state` | 42 |
- 5-fold cross-validation

### 4. Neural Network (Keras)
| Parameter | Value |
|-----------|-------|
| Architecture | Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(1) |
| Activation | ReLU (hidden layers), Linear (output) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | MSE |
| Epochs | 100 |
| Batch size | 32 |
| Validation split | 10% of training data |
| Dropout rate | 0.2 |
- 3-fold cross-validation (fewer folds to reduce training time)

### Preprocessing
- **Numeric features**: Area (sq. yards), Beds, Baths, Price per sq. yard — StandardScaler normalization
- **Categorical features**: Property Type, Location — OneHotEncoder (`handle_unknown="ignore"`)
- **Outlier removal**: IQR method (1.5× interquartile range)

### Evaluation Metrics
- **R²** (coefficient of determination) — primary selection metric
- **RMSE** (root mean squared error)
- **MAE** (mean absolute error)

The model with the highest R² on the test set is automatically selected and saved.

## Data Collection

The scraper collects property listings from Zameen.com:

```bash
python webscraping.py
```

## Project Structure

```
├── api.py                 # FastAPI prediction API
├── dashboard.py           # Streamlit interactive dashboard
├── train.py               # Model training & evaluation CLI
├── webscraping.py         # Zameen.com property scraper
├── src/
│   ├── config.py          # Paths and constants
│   ├── preprocessing.py   # Data cleaning & feature engineering
│   └── model.py           # Model training, evaluation, persistence
├── data/
│   └── property-data.csv  # Raw scraped data
├── models/                # Saved model artifacts (gitignored)
├── requirements.txt
└── Dockerfile
```
