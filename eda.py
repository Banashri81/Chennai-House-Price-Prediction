
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/cleaned_chennai_house_price.csv")
OUTPUT_DIR = Path("results/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


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

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())


# ==========================================
# 3. DESCRIPTIVE STATISTICS
# ==========================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(df.describe())

print("\nCategorical statistics:")
print(df.describe(include="object"))

df.describe().to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUE ANALYSIS ==========")

print(df.isnull().sum())

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values in Chennai Housing Dataset")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "missing_values.png",
    dpi=300
)

plt.show()


# ==========================================
# 5. SALES PRICE DISTRIBUTION
# ==========================================

print("\n========== SALES PRICE DISTRIBUTION ==========")

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="SALES_PRICE",
    kde=True,
    bins=30
)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sales Price (INR)")
plt.ylabel("Number of Properties")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "price_distribution.png",
    dpi=300
)

plt.show()


# ==========================================
# 6. PROPERTY AREA VS SALES PRICE
# ==========================================

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="INT_SQFT",
    y="SALES_PRICE",
    alpha=0.6
)

plt.title("Property Area vs Sales Price")
plt.xlabel("Interior Area (Square Feet)")
plt.ylabel("Sales Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "area_vs_price.png",
    dpi=300
)

plt.show()


# ==========================================
# 7. LOCALITY PRICE ANALYSIS
# ==========================================

print("\n========== LOCALITY ANALYSIS ==========")

locality_prices = (
    df.groupby("AREA")["SALES_PRICE"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)

print(locality_prices)

locality_prices.to_csv(
    OUTPUT_DIR / "locality_analysis.csv"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="AREA",
    y="SALES_PRICE",
    estimator="mean",
    errorbar=None
)

plt.title("Average House Price by Locality")
plt.xlabel("Locality")
plt.ylabel("Average Sales Price (INR)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "locality_price.png",
    dpi=300
)

plt.show()


# ==========================================
# 8. BEDROOM ANALYSIS
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="N_BEDROOM",
    y="SALES_PRICE"
)

plt.title("Number of Bedrooms vs Sales Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Sales Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "bedroom_vs_price.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. BUILDING TYPE ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="BUILDTYPE",
    y="SALES_PRICE"
)

plt.title("Building Type vs Sales Price")
plt.xlabel("Building Type")
plt.ylabel("Sales Price (INR)")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "building_type_analysis.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. PARKING FACILITY ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PARK_FACIL",
    y="SALES_PRICE"
)

plt.title("Parking Facility vs Sales Price")
plt.xlabel("Parking Facility")
plt.ylabel("Sales Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "parking_analysis.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. CORRELATION ANALYSIS
# ==========================================

print("\n========== CORRELATION MATRIX ==========")

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

print(correlation)

correlation.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ==========================================
# 12. DISTANCE FROM MAIN ROAD ANALYSIS
# ==========================================

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="DIST_MAINROAD",
    y="SALES_PRICE"
)

plt.title("Distance from Main Road vs Sales Price")
plt.xlabel("Distance from Main Road")
plt.ylabel("Sales Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distance_vs_price.png",
    dpi=300
)

plt.show()


# ==========================================
# 13. FINAL SUMMARY
# ==========================================

print("\n========== EDA COMPLETED ==========")

print("Total records:", df.shape[0])
print("Total features:", df.shape[1])

print("Average house price:", df["SALES_PRICE"].mean())

print("Minimum house price:", df["SALES_PRICE"].min())

print("Maximum house price:", df["SALES_PRICE"].max())

print("\nAll graphs and reports saved in:")
print(OUTPUT_DIR.resolve())

print("\nEDA EXECUTION SUCCESSFUL!")
