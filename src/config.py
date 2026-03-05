from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "property-data.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Locations with fewer samples than this fall back to parent locality
MIN_LOCATION_SAMPLES = 10

FEATURE_COLS = ["Area", "Beds", "Baths", "area_per_bed"]
CATEGORICAL_COLS = ["Type", "Location_Clean"]
TARGET_COL = "Price"
