
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.metrics import r2_score

# Load prediction results
file_path = Path(
    "results/model_evaluation/tuned_model_predictions.csv"
)

df = pd.read_csv(file_path)

actual = df["Actual_Price"]
predicted = df["Predicted_Price"]

# Create output directory
output_dir = Path("results/model_interpretation")
output_dir.mkdir(parents=True, exist_ok=True)

# ==================================
# 1. ACTUAL VS PREDICTED
# ==================================

plt.figure(figsize=(9, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.6
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    output_dir / "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ==================================
# 2. RESIDUAL ANALYSIS
# ==================================

residuals = actual - predicted

plt.figure(figsize=(9, 6))

plt.scatter(
    predicted,
    residuals,
    alpha=0.6
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted House Price")
plt.ylabel("Residual Error")
plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    output_dir / "residual_analysis.png",
    dpi=300
)

plt.show()

# ==================================
# 3. ERROR DISTRIBUTION
# ==================================

plt.figure(figsize=(9, 6))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    output_dir / "error_distribution.png",
    dpi=300
)

plt.show()

# ==================================
# 4. EVALUATION SUMMARY
# ==================================

print("\nMODEL VISUALIZATION SUMMARY")

print("R2 Score:", round(
    r2_score(actual, predicted), 4
))

print("Mean Absolute Error:", round(
    np.mean(np.abs(residuals)), 2
))

print("Mean Residual:", round(
    residuals.mean(), 2
))

print("\nVisualizations saved successfully!")
