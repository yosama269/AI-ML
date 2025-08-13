import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")
print(f"Dataset loaded with shape: {df.shape}")

# Clean columns
df.columns = df.columns.str.strip()

# Drop missing targets
df = df.dropna(subset=['loan_status'])

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category').cat.codes)

# Features and target
X = df.drop(columns=['loan_status', 'loan_id'])
y = df['loan_status']

# Tune Decision Tree helper
def tune_decision_tree(X_train, y_train, X_val, y_val, max_depth_values):
    best_depth = None
    best_val_acc = 0
    for depth in max_depth_values:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_val_pred = dt.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_depth = depth
    return best_depth

# Step 1: Tune Random Forest once on one split
print("Tuning Random Forest hyperparameters...")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rf_random_search.fit(X_train_res, y_train_res)

best_params = rf_random_search.best_params_
print("Best RF params:", best_params)
print("Best RF F1 score (CV):", rf_random_search.best_score_)

# Step 2: Evaluate models with multiple splits using best RF params
n_splits = 10

log_train_accs, log_test_accs = [], []
dt_train_accs, dt_test_accs = [], []
rf_train_accs, rf_test_accs = [], []

final_log_reg, final_dt, final_rf = None, None, None
final_X_test, final_y_test = None, None

for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42 + i)

    smote = SMOTE(random_state=42)
    X_train_sub_res, y_train_sub_res = smote.fit_resample(X_train_sub, y_train_sub)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_res, y_train_res)

    # Tune Decision Tree
    max_depth_values = [2, 4, 6, 8, 10, None]
    best_depth = tune_decision_tree(X_train_sub_res, y_train_sub_res, X_val, y_val, max_depth_values)
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt.fit(X_train_res, y_train_res)

    # Random Forest with tuned params
    rf_tuned = RandomForestClassifier(**best_params, random_state=42)
    rf_tuned.fit(X_train_res, y_train_res)

    # Evaluate on original train/test sets
    log_train_accs.append(accuracy_score(y_train, log_reg.predict(X_train)))
    log_test_accs.append(accuracy_score(y_test, log_reg.predict(X_test)))

    dt_train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    dt_test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

    rf_train_accs.append(accuracy_score(y_train, rf_tuned.predict(X_train)))
    rf_test_accs.append(accuracy_score(y_test, rf_tuned.predict(X_test)))

    # Save last split models and data
    final_log_reg, final_dt, final_rf = log_reg, dt, rf_tuned
    final_X_test, final_y_test = X_test, y_test

print("\nResults after running multiple random splits with SMOTE and tuned Random Forest:\n")

print("Logistic Regression:")
print(f"- Average training accuracy: {np.mean(log_train_accs)*100:.1f}%")
print(f"- Training accuracy std dev: {np.std(log_train_accs)*100:.1f}%")
print(f"- Average test accuracy: {np.mean(log_test_accs)*100:.1f}%")
print(f"- Test accuracy std dev: {np.std(log_test_accs)*100:.1f}%\n")

print("Decision Tree (tuned max_depth):")
print(f"- Average training accuracy: {np.mean(dt_train_accs)*100:.1f}%")
print(f"- Training accuracy std dev: {np.std(dt_train_accs)*100:.1f}%")
print(f"- Average test accuracy: {np.mean(dt_test_accs)*100:.1f}%")
print(f"- Test accuracy std dev: {np.std(dt_test_accs)*100:.1f}%\n")

print("Random Forest (tuned):")
print(f"- Average training accuracy: {np.mean(rf_train_accs)*100:.1f}%")
print(f"- Training accuracy std dev: {np.std(rf_train_accs)*100:.1f}%")
print(f"- Average test accuracy: {np.mean(rf_test_accs)*100:.1f}%")
print(f"- Test accuracy std dev: {np.std(rf_test_accs)*100:.1f}%\n")

print("\nClassification Report for Logistic Regression:")
print(classification_report(final_y_test, final_log_reg.predict(final_X_test)))

print("\nClassification Report for Decision Tree (tuned max_depth):")
print(classification_report(final_y_test, final_dt.predict(final_X_test)))

print("\nClassification Report for Random Forest (tuned):")
print(classification_report(final_y_test, final_rf.predict(final_X_test)))

# Plot confusion matrices
y_test_pred_lr = final_log_reg.predict(final_X_test)
y_test_pred_dt = final_dt.predict(final_X_test)
y_test_pred_rf = final_rf.predict(final_X_test)

cm_lr = confusion_matrix(final_y_test, y_test_pred_lr)
cm_dt = confusion_matrix(final_y_test, y_test_pred_dt)
cm_rf = confusion_matrix(final_y_test, y_test_pred_rf)

acc_lr = accuracy_score(final_y_test, y_test_pred_lr)
acc_dt = accuracy_score(final_y_test, y_test_pred_dt)
acc_rf = accuracy_score(final_y_test, y_test_pred_rf)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ConfusionMatrixDisplay(cm_lr).plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
axes[0].set_title(f'Logistic Regression\nAccuracy: {acc_lr:.3f}')

ConfusionMatrixDisplay(cm_dt).plot(ax=axes[1], cmap=plt.cm.Greens, colorbar=False)
axes[1].set_title(f'Decision Tree (tuned max_depth)\nAccuracy: {acc_dt:.3f}')

ConfusionMatrixDisplay(cm_rf).plot(ax=axes[2], cmap=plt.cm.Purples, colorbar=False)
axes[2].set_title(f'Random Forest (tuned)\nAccuracy: {acc_rf:.3f}')

plt.suptitle('Comparison of Confusion Matrices for Models with SMOTE and Tuned RF', fontsize=16)
plt.tight_layout()
plt.show()
