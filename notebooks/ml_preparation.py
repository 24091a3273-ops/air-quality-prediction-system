import pandas as pd
from sklearn.model_selection import train_test_split

# Load AQI dataset
file_path = "dataset/aqi_air_quality.csv"
df = pd.read_csv(file_path)

print("AQI dataset loaded successfully!")

# Features used by the ML model
features = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "C6H6(GT)",
    "T",
    "RH",
    "AH"
]

target = "AQI"

# Select features and target
X = df[features]
y = df[target]

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nML data preparation completed successfully!")