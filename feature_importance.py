
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load optimized model
MODEL_PATH = "models/tuned_random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

# Extract feature importance
if hasattr(model, "feature_importances_"):
    estimator = model
    feature_names = getattr(
        model, "feature_names_in_", None
    )

elif hasattr(model, "named_steps"):
    estimator = None

    for step in reversed(list(model.named_steps.values())):
        if hasattr(step, "feature_importances_"):
            estimator = step
            break

    if estimator is None:
        raise ValueError("Random Forest estimator not found.")

    try:
        feature_names = model[:-1].get_feature_names_out()
    except Exception:
        feature_names = getattr(
            model, "feature_names_in_", None
        )

else:
    raise ValueError("Unsupported model format.")

if feature_names is None:
    raise ValueError(
        "Feature names unavailable. Check your training pipeline."
    )

importance = estimator.feature_importances_

if len(feature_names) != len(importance):
    raise ValueError(
        "Feature count mismatch. Check preprocessing."
    )

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

# Save results
os.makedirs("results/model_interpretation", exist_ok=True)

importance_df.to_csv(
    "results/model_interpretation/feature_importance.csv",
    index=False
)

# Plot top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Features Affecting House Prices")

plt.tight_layout()

plt.savefig(
    "results/model_interpretation/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed!")
