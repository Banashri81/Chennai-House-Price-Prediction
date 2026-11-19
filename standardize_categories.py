
import pandas as pd

# Load standardized dataset
file_path = r"data/standardized_chennai_house_price.csv"
df = pd.read_csv(file_path)

# Category correction mappings
corrections = {
    "AREA": {
        "Adyr": "Adyar",
        "Ana Nagar": "Anna Nagar",
        "Ann Nagar": "Anna Nagar",
        "Chormpet": "Chrompet",
        "Chrmpet": "Chrompet",
        "Chrompt": "Chrompet",
        "KKNagar": "KK Nagar",
        "Karapakam": "Karapakkam",
        "TNagar": "T Nagar",
        "Velchery": "Velachery"
    },

    "BUILDTYPE": {
        "Comercial": "Commercial",
        "Other": "Others"
    },

    "PARK_FACIL": {
        "Noo": "No"
    },

    "STREET": {
        "NoAccess": "No Access",
        "Pavd": "Paved"
    },

    "UTILITY_AVAIL": {
        "All Pub": "AllPub"
    }
}

# Apply corrections
for column, mapping in corrections.items():
    df[column] = df[column].replace(mapping)

# Save standardized dataset
output_path = r"data/standardized_chennai_house_price.csv"
df.to_csv(output_path, index=False)

# Display category counts after correction
for column in corrections:
    print("\n" + "=" * 45)
    print(f"STANDARDIZED VALUES: {column}")
    print("=" * 45)
    print(df[column].value_counts(dropna=False))

print("\nStandardization completed successfully.")
print("Dataset shape:", df.shape)
print("Saved to:", output_path)
