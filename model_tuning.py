
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/standardized_chennai_house_price.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/model_evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "SALES_PRICE"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

df = df.drop_duplicates().copy()

df[TARGET] = pd.to_numeric(
    df[TARGET], errors="coerce"
)

df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0].copy()


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

print("\n========== FEATURE ENGINEERING ==========")

df["DATE_BUILD"] = pd.to_datetime(
    df["DATE_BUILD"],
    errors="coerce",
    dayfirst=True
)

df["BUILD_YEAR"] = df["DATE_BUILD"].dt.year

df["BUILDING_AGE"] = (
    datetime.now().year - df["BUILD_YEAR"]
)

df.loc[
    (df["BUILDING_AGE"] < 0) |
    (df["BUILDING_AGE"] > 200),
    "BUILDING_AGE"
] = np.nan

df.drop(
    columns=["DATE_BUILD", "BUILD_YEAR"],
    inplace=True
)

print("Building age feature created.")


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "numerical",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. RANDOM FOREST PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    )
])


# ==========================================
# 8. HYPERPARAMETER SEARCH SPACE
# ==========================================

param_distributions = {

    "model__n_estimators": [
        100, 200, 300, 400, 500
    ],

    "model__max_depth": [
        None, 10, 15, 20, 25, 30
    ],

    "model__min_samples_split": [
        2, 5, 10, 15
    ],

    "model__min_samples_leaf": [
        1, 2, 4, 6
    ],

    "model__max_features": [
        "sqrt", 0.7, 1.0
    ]
}


# ==========================================
# 9. RANDOMIZED SEARCH CV
# ==========================================

print("\n========== HYPERPARAMETER TUNING ==========")

cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=cv_strategy,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
    refit=True
)

search.fit(X_train, y_train)

print("\n========== TUNING COMPLETED ==========")

print("Best parameters:")
print(search.best_params_)

print(
    "Best cross-validation RMSE:",
    round(-search.best_score_, 2)
)


# ==========================================
# 10. BEST MODEL EVALUATION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

best_model = search.best_estimator_

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

train_r2 = r2_score(
    y_train, train_predictions
)

test_r2 = r2_score(
    y_test, test_predictions
)

mae = mean_absolute_error(
    y_test, test_predictions
)

mse = mean_squared_error(
    y_test, test_predictions
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    y_test, test_predictions
) * 100

print("\nTraining R2:", round(train_r2, 4))
print("Testing R2:", round(test_r2, 4))
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")

print(
    "Train-Test R2 gap:",
    round(train_r2 - test_r2, 4)
)


# ==========================================
# 11. SAVE OPTIMIZED MODEL
# ==========================================

print("\n========== SAVING OPTIMIZED MODEL ==========")

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_random_forest_model.pkl"
)

# Update the main application model

joblib.dump(
    best_model,
    MODEL_DIR / "price_model.pkl"
)

print("Optimized model saved successfully.")


# ==========================================
# 12. SAVE TUNING RESULTS
# ==========================================

tuning_results = pd.DataFrame(search.cv_results_)

tuning_results.to_csv(
    RESULT_DIR / "hyperparameter_results.csv",
    index=False
)

metrics = {
    "model": "Tuned Random Forest",
    "best_parameters": search.best_params_,
    "cross_validation_rmse": float(-search.best_score_),
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "mape_percentage": float(mape),
    "train_test_r2_gap": float(train_r2 - test_r2)
}

with open(
    MODEL_DIR / "tuned_model_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)


# ==========================================
# 13. SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({

    "Actual_Price": y_test.values,

    "Predicted_Price": test_predictions,

    "Prediction_Error": (
        y_test.values - test_predictions
    )
})

prediction_df.to_csv(
    RESULT_DIR / "tuned_model_predictions.csv",
    index=False
)


# ==========================================
# 14. FINAL REPORT
# ==========================================

print("\n====================================")
print("HYPERPARAMETER TUNING COMPLETED")
print("====================================")

print("Best CV RMSE:", round(-search.best_score_, 2))
print("Test R2:", round(test_r2, 4))
print("Test RMSE:", round(rmse, 2))

print("\nGenerated files:")

print("1. tuned_random_forest_model.pkl")
print("2. price_model.pkl")
print("3. tuned_model_metrics.json")
print("4. hyperparameter_results.csv")
print("5. tuned_model_predictions.csv")

print("\nALL RESULTS SAVED SUCCESSFULLY!")
