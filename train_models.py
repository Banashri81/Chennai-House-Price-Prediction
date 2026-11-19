
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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

TARGET = "SALES_PRICE"

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 3. DATA VALIDATION
# ==========================================

print("\n========== DATA VALIDATION ==========")

if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET} not found.")

df = df.drop_duplicates().copy()

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(subset=[TARGET])

df = df[df[TARGET] > 0].copy()

print("Records after validation:", len(df))


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

print("\n========== FEATURE ENGINEERING ==========")

# Convert DATE_BUILD into datetime

df["DATE_BUILD"] = pd.to_datetime(
    df["DATE_BUILD"],
    errors="coerce",
    dayfirst=True
)

# Extract building year

df["BUILD_YEAR"] = df["DATE_BUILD"].dt.year

# Calculate building age

CURRENT_YEAR = datetime.now().year

df["BUILDING_AGE"] = (
    CURRENT_YEAR - df["BUILD_YEAR"]
)

# Remove original date column

df.drop(columns=["DATE_BUILD"], inplace=True)

# Remove invalid building ages

df.loc[
    (df["BUILDING_AGE"] < 0) |
    (df["BUILDING_AGE"] > 200),
    "BUILDING_AGE"
] = np.nan

df.drop(columns=["BUILD_YEAR"], inplace=True)

print("New feature created: BUILDING_AGE")

print("\nFeatures:")
print(df.columns.tolist())


# ==========================================
# 5. DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify numerical and categorical columns

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================
# 6. PREPROCESSING PIPELINES
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 7. TRAIN-TEST SPLIT
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
# 8. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}


# ==========================================
# 9. MODEL TRAINING AND EVALUATION
# ==========================================

print("\n========== MODEL TRAINING ==========")

results = []
trained_models = {}

for name, algorithm in models.items():

    print("\n--------------------------------")
    print("Training:", name)
    print("--------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", algorithm)
        ]
    )

    # Train model

    pipeline.fit(X_train, y_train)

    # Predictions

    train_predictions = pipeline.predict(X_train)
    predictions = pipeline.predict(X_test)

    # Evaluation metrics

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    ) * 100

    # Store results

    results.append({

        "Model": name,
        "Train_R2": train_r2,
        "Test_R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_Percentage": mape

    })

    trained_models[name] = pipeline

    # Save individual model

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + "_model.pkl"
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )

    print("Train R2:", round(train_r2, 4))
    print("Test R2:", round(r2, 4))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

print("\n========== BEST MODEL SELECTION ==========")

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_rmse = results_df.iloc[0]["RMSE"]

best_r2 = results_df.iloc[0]["Test_R2"]

# Save best model using the filename
# expected by the application

joblib.dump(
    best_model,
    MODEL_DIR / "price_model.pkl"
)

print("Best model:", best_model_name)
print("Best RMSE:", round(best_rmse, 2))
print("Best R2:", round(best_r2, 4))


# ==========================================
# 12. SAVE MODEL METRICS
# ==========================================

metrics = {
    "best_model": best_model_name,
    "best_rmse": float(best_rmse),
    "best_r2": float(best_r2),
    "models": results_df.to_dict(orient="records")
}

with open(
    MODEL_DIR / "model_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================
# 13. SAVE TEST PREDICTIONS
# ==========================================

best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({

    "Actual_Price": y_test.values,

    "Predicted_Price": best_predictions,

    "Prediction_Error": (
        y_test.values - best_predictions
    )

})

prediction_results.to_csv(
    RESULT_DIR / "best_model_predictions.csv",
    index=False
)


# ==========================================
# 14. FINAL REPORT
# ==========================================

print("\n======================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("======================================")

print("\nTotal records:", len(df))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print("\nBest performing model:", best_model_name)

print("Best R2:", round(best_r2, 4))

print("Best RMSE:", round(best_rmse, 2))

print("\nSaved files:")

print("1. Individual trained models:", MODEL_DIR.resolve())

print("2. Best model: models/price_model.pkl")

print("3. Model metrics: models/model_metrics.json")

print("4. Model comparison:",
      RESULT_DIR / "model_comparison.csv")

print("5. Predictions:",
      RESULT_DIR / "best_model_predictions.csv")

print("\nALL MODELS TRAINED SUCCESSFULLY!")
