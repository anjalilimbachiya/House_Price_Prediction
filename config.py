from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "House Price Prediction Dataset.csv"
MODEL_PATH = MODELS_DIR / "house_price_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "metrics.csv"

DATASET_REMOTE_URL = (
    "https://raw.githubusercontent.com/nikhilkhosare/house-price-prediction/"
    "main/House_Price_Prediction_Dataset.csv"
)

TARGET_COLUMN = "Price"
ID_COLUMN = "Id"

NUMERIC_FEATURES = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
]

CATEGORICAL_FEATURES = [
    "Location",
    "Condition",
    "Garage",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CURRENCY_TEXT = (
    "Prediction uses the same price unit/currency represented by the dataset. "
    "The value is shown in a clean currency-style format."
)
