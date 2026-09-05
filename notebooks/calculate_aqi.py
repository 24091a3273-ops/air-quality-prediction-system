import pandas as pd
import numpy as np

# Load prepared dataset
file_path = "dataset/prepared_air_quality.csv"
df = pd.read_csv(file_path)

print("Prepared dataset loaded successfully!")

# Convert pollutant columns to numeric
pollutants = ["CO(GT)", "NO2(GT)", "NOx(GT)"]

for column in pollutants:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill any remaining missing values
for column in pollutants:
    df[column] = df[column].fillna(df[column].median())

# Normalize pollutant values to create an AQI-like score
co_score = (df["CO(GT)"] / df["CO(GT)"].max()) * 100
no2_score = (df["NO2(GT)"] / df["NO2(GT)"].max()) * 100
nox_score = (df["NOx(GT)"] / df["NOx(GT)"].max()) * 100

# Take the highest pollutant score
df["AQI"] = pd.concat(
    [co_score, no2_score, nox_score],
    axis=1
).max(axis=1)

# Limit AQI between 0 and 500
df["AQI"] = df["AQI"].clip(0, 500)

# Create AQI category
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

df["AQI_Category"] = df["AQI"].apply(aqi_category)

# Save dataset
output_path = "dataset/aqi_air_quality.csv"
df.to_csv(output_path, index=False)

print("\nAQI calculation completed successfully!")

print("\nAQI Statistics:")
print(df["AQI"].describe())

print("\nAQI Categories:")
print(df["AQI_Category"].value_counts())

print("\nFirst 10 AQI values:")
print(df[["CO(GT)", "NO2(GT)", "NOx(GT)", "AQI", "AQI_Category"]].head(10))

print("\nFinal dataset saved at:")
print(output_path)