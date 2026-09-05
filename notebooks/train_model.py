import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load AQI dataset
file_path = "dataset/aqi_air_quality.csv"
df = pd.read_csv(file_path)

print("AQI dataset loaded successfully!")

# Features
features = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "C6H6(GT)",
    "T",
    "RH",
    "AH"
]

# Target
target = "AQI"

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Random Forest model...")

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
print("=======================================")

# Save model
model_path = "model/aqi_random_forest.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)

# Show some predictions
results = pd.DataFrame({
    "Actual AQI": y_test.values[:10],
    "Predicted AQI": y_pred[:10]
})

print("\nSample Predictions:")
print(results)