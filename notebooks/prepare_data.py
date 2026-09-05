import pandas as pd
import numpy as np

# Load cleaned dataset
file_path = "dataset/cleaned_air_quality.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Convert all possible numeric columns to numeric
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nMissing values BEFORE handling:")
print(df.isnull().sum())

# Fill missing numerical values using the median
for column in df.select_dtypes(include=np.number).columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values AFTER handling:")
print(df.isnull().sum())

# Save prepared dataset
output_path = "dataset/prepared_air_quality.csv"

df.to_csv(output_path, index=False)

print("\nData preparation completed successfully!")
print("Prepared dataset saved at:")
print(output_path)