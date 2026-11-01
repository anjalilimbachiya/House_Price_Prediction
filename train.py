import warnings
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    CATEGORICAL_FEATURES,
    DATASET_REMOTE_URL,
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore")


def ensure_directories():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset_if_missing(path: Path = DATA_PATH):
    ensure_directories()
    if path.exists() and path.stat().st_size > 200:
        return path

    try:
        urlretrieve(DATASET_REMOTE_URL, str(path))
        return path
    except Exception:
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Dataset could not be downloaded automatically. "
            f"Please place the CSV file at: {path}"
        )


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    download_dataset_if_missing(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place the CSV file in data/."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty. Please check the CSV file.")

    required_columns = [
        "Id",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "YearBuilt",
        "Location",
        "Condition",
        "Garage",
        "Price",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {missing_columns}"
        )

    return df


def prepare_data(df: pd.DataFrame):
    working_df = df.copy()
    working_df = working_df.drop_duplicates().reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        working_df[col] = working_df[col].astype(str).str.strip()
        working_df[col] = working_df[col].replace({
            "nan": np.nan,
            "None": np.nan,
            "": np.nan,
            "N/A": np.nan,
        })

    for col in NUMERIC_FEATURES:
        median_value = working_df[col].median()
        working_df[col] = working_df[col].fillna(median_value)

    for col in CATEGORICAL_FEATURES:
        mode_value = working_df[col].mode(dropna=True)
        if not mode_value.empty:
            working_df[col] = working_df[col].fillna(mode_value.iloc[0])
        else:
            working_df[col] = working_df[col].fillna("Unknown")

    if TARGET_COLUMN not in working_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    X = working_df[FEATURE_COLUMNS].copy()
    y = working_df[TARGET_COLUMN].astype(float).copy()
    return X, y


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def evaluate_models(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=20
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=20),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=20),
        "Gradient Boosting": GradientBoostingRegressor(random_state=20),
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_models(pipeline, X_test, y_test)

        entry = {
            "Model": model_name,
            "MAE": round(metrics["MAE"], 4),
            "MSE": round(metrics["MSE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "R2": round(metrics["R2"], 4),
        }

        results.append(entry)
        trained_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results)

    best_row = results_df.iloc[0].copy()
    for row in results_df.iloc[1:].itertuples(index=False):
        row_dict = row._asdict()
        current_r2 = float(row_dict["R2"])
        current_rmse = float(row_dict["RMSE"])
        best_r2 = float(best_row["R2"])
        best_rmse = float(best_row["RMSE"])

        if current_r2 > best_r2 + 1e-4:
            best_row = row_dict
        elif abs(current_r2 - best_r2) <= 0.01 and current_rmse < best_rmse:
            best_row = row_dict

    best_model_name = str(best_row["Model"])
    best_pipeline = trained_pipelines[best_model_name]

    return results_df, best_model_name, best_pipeline, trained_pipelines


def save_model(model, path: Path = MODEL_PATH):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(model.named_steps["preprocessor"], PREPROCESSOR_PATH)
    return path, PREPROCESSOR_PATH


def save_metrics(results_df: pd.DataFrame, path: Path = METRICS_PATH):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    return path


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Please run python train.py first.")
    return joblib.load(path)


def predict_price(input_df: pd.DataFrame):
    model = load_model()

    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            raise ValueError(f"Missing required input column: {col}")

    processed = input_df[FEATURE_COLUMNS].copy()
    prediction = model.predict(processed)[0]
    return float(prediction)
