import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Đọc dữ liệu
DATA_PATH = Path("diabetes.csv")
assert DATA_PATH.exists(), "Không tìm thấy file diabetes.csv trong thư mục hiện tại!"
df = pd.read_csv(DATA_PATH)

# 2. Xác định cột mục tiêu (target)
def infer_target_column(df: pd.DataFrame) -> str:
    preferred = [
        "Outcome", "outcome", "Diabetes", "diabetes",
        "Class", "class", "Target", "target", "Label", "label"
    ]
    for c in preferred:
        if c in df.columns:
            return c
    last_col = df.columns[-1]
    uniq = df[last_col].dropna().unique()
    if len(uniq) <= 2:
        return last_col
    for c in df.columns:
        if len(df[c].dropna().unique()) <= 2:
            return c
    return df.columns[-1]

target_col = infer_target_column(df)
feature_cols = [c for c in df.columns if c != target_col]

# 3. Xử lý dữ liệu: thay 0 thành NaN
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_fixed = df.copy()
for c in zero_as_missing:
    if c in df_fixed.columns and np.issubdtype(df_fixed[c].dtype, np.number):
        df_fixed.loc[df_fixed[c] == 0, c] = np.nan

# 4. Tách cột numeric / categorical
X_all = df_fixed[feature_cols]
y_all = df_fixed[target_col]
numeric_features = X_all.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in feature_cols if c not in numeric_features]

# 5. Chia dữ liệu train/test
stratify = y_all if pd.Series(y_all).nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=stratify
)

# 6. Pipeline tiền xử lý và mô hình
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
])

# 7. GridSearchCV tối ưu tham số
param_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# 8. Đánh giá mô hình
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
else:
    y_proba = best_model.decision_function(X_test)

y_pred = (y_proba >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "average_precision": float(average_precision_score(y_test, y_proba)),
}
conf_matrix = confusion_matrix(y_test, y_pred)

# 9. Lưu kết quả và mô hình
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report_df.to_csv("report.csv", index=True)

bundle = {
    "model": best_model,
    "feature_cols": feature_cols,
    "target_col": target_col,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": metrics,
}
joblib.dump(bundle, "diabetes_model.joblib")

# 10. Hiển thị kết quả
print("\nTRAINING DONE")
print(f"Target: {target_col}")
print(f"Features: {feature_cols}")
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nMetrics:")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("\nSaved: diabetes_model.joblib, report.csv")

# 11. Vẽ Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dự đoán 0", "Dự đoán 1"],
            yticklabels=["Thực tế 0", "Thực tế 1"])
plt.xlabel("Kết quả dự đoán")
plt.ylabel("Giá trị thực tế")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()
