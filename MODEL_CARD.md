# House Price Prediction System

A professional machine learning application for estimating house prices using property features such as area, bedrooms, bathrooms, floors, year built, location, condition, and garage availability.

## Overview

This project combines:
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

The application trains several regression models, compares them, selects the best one, and provides a premium AI dashboard for live predictions.

## Dataset

The real dataset is downloaded automatically if it is missing:

https://raw.githubusercontent.com/nikhilkhosare/house-price-prediction/main/House_Price_Prediction_Dataset.csv

Expected columns:
- Id
- Area
- Bedrooms
- Bathrooms
- Floors
- YearBuilt
- Location
- Condition
- Garage
- Price

Target:
- Price

Important:
- Id is not used as a prediction feature.
- Price is never included as an input feature.

## Features

- Area
- Bedrooms
- Bathrooms
- Floors
- YearBuilt
- Location
- Condition
- Garage

## Machine Learning Models

The app compares:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Preprocessing

- Numeric features: median imputation + StandardScaler
- Categorical features: most frequent imputation + OneHotEncoder(handle_unknown="ignore")
- ColumnTransformer used for end-to-end preprocessing

## Evaluation Metrics

- MAE
- MSE
- RMSE
- R²

## Project Structure

House_Price_Prediction/
├── app.py
├── train.py
├── model.py
├── config.py
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── data/
│   └── House Price Prediction Dataset.csv
├── models/
│   ├── house_price_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.csv
└── assets/

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Run the Application

```bash
streamlit run app.py
```

## Prediction Workflow

1. User enters property details.
2. The saved preprocessing and model pipeline transforms the inputs.
3. The trained model predicts the probable house price.
4. The frontend displays the result in a premium dashboard card.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Joblib
