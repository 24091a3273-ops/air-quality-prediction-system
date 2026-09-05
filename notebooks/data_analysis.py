import pandas as pd
import numpy as np

# Load cleaned dataset
file_path = "dataset/cleaned_air_quality.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully!")

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())