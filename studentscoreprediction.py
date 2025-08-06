import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('student_performance.csv')
# STEP 1: Check for correlation using a heatmap


# Create the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Selected features and target
features = [
    'study_hours_per_day',
    'attendance_percentage',
    'sleep_hours',
    'exercise_frequency',
    'mental_health_rating'
]
target = 'exam_score'

# Drop missing values in the selected columns
df = df.dropna(subset=features + [target])

# Features (X) and Target (y)
X = df[features]
y = df[target]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print("Improved Linear Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization: Actual vs Predicted with y = x and regression line
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Predicted Points", alpha=0.7)

# Red dashed line (perfect prediction line)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Prediction (y = x)")

# Blue regression line (best fit line through predicted data)
# Fit a line between actual and predicted values
regression_coeffs = np.polyfit(y_test, y_pred, deg=1)
regression_line = np.poly1d(regression_coeffs)
y_fit = regression_line(np.sort(y_test))

plt.plot(np.sort(y_test), y_fit, color='blue', label="Regression Fit")

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.legend()
plt.grid(True)
plt.show()

# === Polynomial Regression (Degree 2) ===

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create a pipeline with PolynomialFeatures + LinearRegression
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Fit the model on training data
poly_model.fit(X_train, y_train)

# Predict on test data
y_poly_pred = poly_model.predict(X_test)

# Evaluate the polynomial model
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_r2 = r2_score(y_test, y_poly_pred)

print("\nPolynomial Regression (Degree 2) Results:")
print(f"Mean Squared Error: {poly_mse:.2f}")
print(f"R² Score: {poly_r2:.2f}")

# Visualization: Actual vs Predicted (Polynomial Regression)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_poly_pred, label="Predicted Points (Poly)", alpha=0.7)

# Add red dashed line: y = x
min_val = min(y_test.min(), y_poly_pred.min())
max_val = max(y_test.max(), y_poly_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Prediction (y = x)")

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores (Polynomial Regression)")
plt.legend()
plt.grid(True)
plt.show()
