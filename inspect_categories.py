import pandas as pd

file_path = r"data/cleaned_chennai_house_price.csv"

df = pd.read_csv(file_path)

categorical_columns = [
    "AREA",
    "BUILDTYPE",
    "PARK_FACIL",
    "STREET",
    "UTILITY_AVAIL"
]

for column in categorical_columns:
    print("\n" + "=" * 50)
    print(f"UNIQUE VALUES IN {column}")
    print("=" * 50)

    for value in sorted(df[column].dropna().astype(str).unique()):
        count = (df[column].astype(str) == value).sum()
        print(f"{value}: {count}")