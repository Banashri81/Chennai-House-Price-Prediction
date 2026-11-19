
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

DATA_PATH = Path("data/Chennai houseing sale.csv")

OUTPUT_PATH = Path("data/cleaned_chennai_house_price.csv")


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET INFORMATION ==========")

    print("Original dataset shape:", df.shape)

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    print("\nColumn names:")
    print(df.columns.tolist())

    return df


# ==========================================
# 3. DATA CLEANING
# ==========================================

def clean_dataset(df):

    print("\n========== DATA CLEANING ==========")

    # Replace invalid text values with NaN
    df = df.replace(
        ["?", "NA", "N/A", "NULL", "null", ""],
        np.nan
    )

    # Remove duplicate records
    duplicate_count = df.duplicated().sum()

    print("Duplicate records:", duplicate_count)

    df = df.drop_duplicates()

    # Numerical columns
    numeric_columns = [
        "INT_SQFT",
        "DIST_MAINROAD",
        "N_BEDROOM",
        "N_BATHROOM",
        "SALES_PRICE"
    ]

    # Convert numerical columns
    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove rows with missing target values
    df = df.dropna(subset=["SALES_PRICE"])

    # Remove invalid target values
    df = df[df["SALES_PRICE"] > 0]

    # Remove invalid property areas
    df = df[df["INT_SQFT"] > 0]

    print("\nMissing values before imputation:")
    print(df.isnull().sum())

    # ======================================
    # 4. HANDLE NUMERICAL MISSING VALUES
    # ======================================

    feature_numeric_columns = [
        "INT_SQFT",
        "DIST_MAINROAD",
        "N_BEDROOM",
        "N_BATHROOM"
    ]

    for col in feature_numeric_columns:

        if col in df.columns:

            median_value = df[col].median()

            df[col] = df[col].fillna(median_value)

    # ======================================
    # 5. HANDLE CATEGORICAL MISSING VALUES
    # ======================================

    categorical_columns = [
        "AREA",
        "PARK_FACIL",
        "BUILDTYPE",
        "UTILITY_AVAIL",
        "STREET"
    ]

    for col in categorical_columns:

        if col in df.columns:

            if df[col].isnull().sum() > 0:

                mode_value = df[col].mode()

                if not mode_value.empty:
                    df[col] = df[col].fillna(
                        mode_value[0]
                    )

    # ======================================
    # 6. CLEAN CATEGORICAL TEXT
    # ======================================

    for col in categorical_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

    # ======================================
    # 7. FINAL VALIDATION
    # ======================================

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    total_missing = df.isnull().sum().sum()

    print("\nTotal missing values:", total_missing)

    if total_missing > 0:
        raise ValueError(
            "Missing values still exist in the dataset."
        )

    print("\nFinal cleaned dataset shape:", df.shape)

    return df


# ==========================================
# 8. SAVE CLEANED DATASET
# ==========================================

def save_dataset(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== DATASET SAVED ==========")

    print("Saved location:", OUTPUT_PATH.resolve())

    print("Total records:", len(df))

    print("Total columns:", len(df.columns))


# ==========================================
# 9. MAIN FUNCTION
# ==========================================

def preprocess():

    df = load_dataset()

    df = clean_dataset(df)

    save_dataset(df)

    return df


if __name__ == "__main__":

    cleaned_df = preprocess()

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
