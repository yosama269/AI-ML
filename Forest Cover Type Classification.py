# Forest Cover Type Classification: Random Forest vs Tuned XGBoost
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("covtype.csv")  # Change path if needed

# Features and target
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# For XGBoost, shift labels to start at 0
y_xgb = y - 1  

# ----------------------------
# 2. Train/Test Split
# ----------------------------
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

# ----------------------------
# 3. Random Forest (default params)
# ----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test_rf)

rf_acc = accuracy_score(y_test_rf, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ----------------------------
# 4. XGBoost with Hyperparameter Tuning (Grid Search)
# ----------------------------
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [4, 6, 8]
}

xgb_base = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective="multi:softmax",
    num_class=7
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2
)

grid_search.fit(X_train_xgb, y_train_xgb)

print(f"Best XGBoost Params: {grid_search.best_params_}")
xgb_best = grid_search.best_estimator_

y_pred_xgb = xgb_best.predict(X_test_xgb) + 1  # Shift back to original labels
xgb_acc = accuracy_score(y_test_rf, y_pred_xgb)
print(f"Tuned XGBoost Accuracy: {xgb_acc:.4f}")

# ----------------------------
# 5. Confusion Matrices
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
sns.heatmap(cm_rf, annot=False, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title(f"Random Forest\nAccuracy: {rf_acc:.4f}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

cm_xgb = confusion_matrix(y_test_rf, y_pred_xgb)
sns.heatmap(cm_xgb, annot=False, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title(f"Tuned XGBoost\nAccuracy: {xgb_acc:.4f}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
