import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset/aqi_air_quality.csv")

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

target = "AQI"

X = df[features]
y = df[target]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Load trained model
model = joblib.load("model/aqi_random_forest.pkl")

# Predict AQI
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("========== MODEL EVALUATION ==========")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
print("======================================")

# Create graph
plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:100],
    label="Actual AQI"
)

plt.plot(
    y_pred[:100],
    label="Predicted AQI"
)

plt.xlabel("Test Sample")
plt.ylabel("AQI")
plt.title("Actual AQI vs Predicted AQI")
plt.legend()

# Save graph
plt.savefig("model/actual_vs_predicted_aqi.png", dpi=300)

plt.show()

print("\nGraph saved successfully!")
print("Location: model/actual_vs_predicted_aqi.png")