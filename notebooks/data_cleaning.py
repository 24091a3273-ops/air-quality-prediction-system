import pandas as pd
import numpy as np

# Load the dataset
file_path = "dataset/AirQualityUCI.csv"

df = pd.read_csv(file_path, sep=";", decimal=",")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Remove completely empty rows
df = df.dropna(axis=0, how="all")

# Replace -200 with NaN
df = df.replace(-200, np.nan)

print("\nMissing values:")
print(df.isnull().sum())

# Save cleaned dataset
output_path = "dataset/cleaned_air_quality.csv"

df.to_csv(output_path, index=False)

print("\nCleaning completed!")
print("Cleaned dataset saved successfully.")