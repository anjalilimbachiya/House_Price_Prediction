import pandas as pd

from config import DATA_PATH, MODEL_PATH, MODELS_DIR
from model import load_data, prepare_data, save_metrics, save_model, train_models


def main():
    print("Loading dataset...")
    try:
        df = load_data(DATA_PATH)
        print(f"Dataset shape: {df.shape}")
    except Exception as exc:
        print(f"Error while loading dataset: {exc}")
        return

    print("Checking missing values...")
    print(df.isnull().sum())

    print("Removing duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset cleaned shape: {df.shape}")

    print("Preparing features and target...")
    X, y = prepare_data(df)

    print("Training regression models...")
    results_df, best_model_name, best_pipeline, _ = train_models(X, y)

    print("\nTraining completed successfully.")
    print()
    for _, row in results_df.iterrows():
        print(
            f"{row['Model']}: "
            f"MAE: {row['MAE']:.4f}, "
            f"MSE: {row['MSE']:.4f}, "
            f"RMSE: {row['RMSE']:.4f}, "
            f"R²: {row['R2']:.4f}"
        )

    print()
    print(f"Best Model: {best_model_name}")

    save_metrics(results_df)
    save_model(best_pipeline)

    print("Saved trained model to:", MODEL_PATH)
    print("Saved preprocessing pipeline to:", MODELS_DIR / "preprocessor.pkl")
    print("Saved evaluation metrics to:", MODELS_DIR / "metrics.csv")


if __name__ == "__main__":
    main()
