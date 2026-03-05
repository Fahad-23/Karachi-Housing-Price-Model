import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACTS_PATH,
    CATEGORICAL_COLS,
    CV_FOLDS,
    FEATURE_COLS,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class KerasRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper around a Keras feedforward network.

    Needed because sklearn's Pipeline expects fit/predict interface,
    and we want cross_val_score to work seamlessly with the neural net.
    """

    def __init__(self, epochs=100, batch_size=32, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.input_dim_ = None

    def _build_model(self, input_dim):
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.input_dim_ = X.shape[1]
        self.model_ = self._build_model(self.input_dim_)
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X, verbose=0).flatten()

    def get_params(self, deep=True):
        return {"epochs": self.epochs, "batch_size": self.batch_size, "verbose": self.verbose}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def _get_candidates() -> dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "NeuralNetwork": KerasRegressor(epochs=100, batch_size=32, verbose=0),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train all candidate models, evaluate on held-out test set, return all pipelines."""
    X = df[FEATURE_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    preprocessor = _build_preprocessor(FEATURE_COLS, CATEGORICAL_COLS)

    results = {}
    pipelines = {}
    best_score = -np.inf
    best_name = None

    for name, model in _get_candidates().items():
        logger.info("Training %s...", name)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Fewer CV folds for neural net to keep training time reasonable
        cv_folds = 3 if name == "NeuralNetwork" else CV_FOLDS
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="r2")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            "r2": round(r2, 4),
            "rmse": round(rmse, 0),
            "mae": round(mae, 0),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
        }
        pipelines[name] = pipeline

        logger.info("%s -- R2=%.4f, RMSE=%.0f, MAE=%.0f, CV R2=%.4f+/-%.4f",
                     name, r2, rmse, mae, cv_scores.mean(), cv_scores.std())

        if r2 > best_score:
            best_score = r2
            best_name = name

    metadata = {
        "best_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "all_results": results,
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "known_types": sorted(df["Type"].unique().tolist()),
        "known_locations": sorted(df["Location_Clean"].unique().tolist()),
        "available_models": list(results.keys()),
    }

    return {
        "best_name": best_name,
        "results": results,
        "pipelines": pipelines,
        "metadata": metadata,
    }


def save_model(pipelines: dict, metadata: dict) -> None:
    """Save all trained pipelines and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Save each pipeline separately so any model can be loaded
    for name, pipeline in pipelines.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, path)
        logger.info("Saved %s to %s", name, path)
    joblib.dump(metadata, ARTIFACTS_PATH)


def load_model(model_name: str = None):
    """Load a specific model pipeline and metadata.

    If model_name is None, loads the best model.
    Returns (pipeline, metadata).
    """
    metadata = joblib.load(ARTIFACTS_PATH)
    if model_name is None:
        model_name = metadata["best_model"]
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {path}")
    pipeline = joblib.load(path)
    return pipeline, metadata


def load_all_models():
    """Load all saved pipelines and metadata. Returns (dict of pipelines, metadata)."""
    metadata = joblib.load(ARTIFACTS_PATH)
    pipelines = {}
    for name in metadata["available_models"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            pipelines[name] = joblib.load(path)
    return pipelines, metadata


def predict(pipeline, property_type: str, location: str, area: float, beds: int, baths: int) -> float:
    area_per_bed = area / max(beds, 1)
    input_df = pd.DataFrame([{
        "Area": area,
        "Beds": beds,
        "Baths": baths,
        "area_per_bed": area_per_bed,
        "Type": property_type,
        "Location_Clean": location,
    }])
    return float(pipeline.predict(input_df)[0])
