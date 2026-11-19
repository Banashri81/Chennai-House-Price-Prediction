
import joblib
import pandas as pd

from database import save_prediction

# Load tuned model
model = joblib.load("models/tuned_random_forest_model.pkl")

# Sample house details
sample_house = pd.DataFrame([{
    "AREA": "Velachery",
    "INT_SQFT": 1200,
    "DIST_MAINROAD": 5,
    "N_BEDROOM": 3,
    "N_BATHROOM": 2,
    "PARK_FACIL": "Yes",
    "BUILDTYPE": "House",
    "UTILITY_AVAIL": "AllPub",
    "STREET": "Paved",
    "BUILDING_AGE": 10
}])

# Predict price
predicted_price = model.predict(sample_house)[0]

# Save prediction to MongoDB
record_id = save_prediction(
    sample_house.iloc[0].to_dict(),
    predicted_price
)

print("\nHOUSE DETAILS")
print(sample_house.to_string(index=False))

print("\nPREDICTED HOUSE PRICE")
print(f"₹{predicted_price:,.2f}")

print("\nMONGODB STATUS")
print("Prediction saved successfully!")
print("Record ID:", record_id)
