
import pandas as pd
import os

file_path = r"D:\real-estate-project\data\Chennai houseing sale.csv"

if os.path.isfile(file_path):

    print("Dataset found successfully!")

    df = pd.read_csv(file_path)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst five records:")
    print(df.head())

else:
    print("Dataset not found!")
    print("Check the file path.")

