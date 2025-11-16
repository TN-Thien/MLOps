import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# 1. Đọc dữ liệu
cols = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

train_path = "./Thuc_hanh_02/Bai_01/datasets/adult.data"
test_path  = "./Thuc_hanh_02/Bai_01/datasets/adult.test"

def load_adult(train_path, test_path):
    df_train = pd.read_csv(train_path, header=None, names=cols, na_values=["?"], skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=cols, na_values=["?"], skipinitialspace=True, skiprows=1)
    df_train["income"] = df_train["income"].str.strip()
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False).str.strip()
    return df_train, df_test

df_train_raw, df_test_raw = load_adult(train_path, test_path)

num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
cat_cols = [c for c in df_train_raw.columns if c not in num_cols + ["income"]]

def binarize_y(y):
    return (y.str.strip() == ">50K").astype(int).to_numpy()

# 2. Chuẩn bị dữ liệu

# Naive
df_train_naive = df_train_raw.dropna().copy()
df_test_naive  = df_test_raw.dropna().copy()

X_train_naive = df_train_naive.drop(columns=["income"])
y_train_naive = binarize_y(df_train_naive["income"])
X_test_naive  = df_test_naive.drop(columns=["income"])
y_test_naive  = binarize_y(df_test_naive["income"])

preprocess_naive = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
])

# Preprocess đầy đủ
X_train_full = df_train_raw.drop(columns=["income"])
y_train_full = binarize_y(df_train_raw["income"])
X_test_full  = df_test_raw.drop(columns=["income"])
y_test_full  = binarize_y(df_test_raw["income"])

numeric_full = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_full = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_full = ColumnTransformer([
    ("num", numeric_full, num_cols),
    ("cat", categorical_full, cat_cols)
])

# 3. Xây dựng mô hình
models = {
    "LogisticRegression": LogisticRegression(max_iter=400, solver="liblinear", class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# 4. Hàm đánh giá mô hình
def evaluate(setting, X_train, y_train, X_test, y_test, preprocessor):
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "Setting": setting, "Model": name, "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1": f1,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
    return pd.DataFrame(rows)

# 5. Chạy huấn luyện & đánh giá
res_naive = evaluate("Naive (không tiền xử lý)", X_train_naive, y_train_naive, X_test_naive, y_test_naive, preprocess_naive)
res_full  = evaluate("Preprocess (đầy đủ)", X_train_full, y_train_full, X_test_full, y_test_full, preprocess_full)

results = pd.concat([res_naive, res_full], ignore_index=True)
print("Kết quả tổng hợp")
print(results.round(4))

# 6. So sánh cải thiện nhờ tiền xử lý
print("\nẢnh hưởng của tiền xử lý (Preprocess - Naive)")
for model in models.keys():
    a = res_naive[res_naive["Model"] == model].iloc[0]
    b = res_full[res_full["Model"] == model].iloc[0]
    print(f"{model:>18s} | Accuracy={b['Accuracy']-a['Accuracy']:+.4f}  "
          f"Precision={b['Precision']-a['Precision']:+.4f}  "
          f"Recall={b['Recall']-a['Recall']:+.4f}  ΔF1={b['F1']-a['F1']:+.4f}")

# 7. Confusion matrix cho mô hình tốt nhất và các mô hình khác
best_model = res_full.sort_values("F1", ascending=False).iloc[0]["Model"]
print(f"\nMô hình tốt nhất (sau tiền xử lý): {best_model}")

pipe_best = Pipeline([("prep", preprocess_full), ("model", models[best_model])])
pipe_best.fit(X_train_full, y_train_full)
y_pred_best = pipe_best.predict(X_test_full)
cm = confusion_matrix(y_test_full, y_pred_best)

plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_model}")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="red")
plt.xticks([0,1], ["<=50K", ">50K"])
plt.yticks([0,1], ["<=50K", ">50K"])
plt.show()

# Thư mục lưu hình
out_dir = Path("./Thuc_hanh_02/Bai_01/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

def plot_cm_save(cm, title, save_path):
    fig = plt.figure(figsize=(5, 4.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    ticks = np.arange(2)
    plt.xticks(ticks, ["<=50K", ">50K"])
    plt.yticks(ticks, ["<=50K", ">50K"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:d}", ha="center", va="center")
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

settings = {
    "Naive (không tiền xử lý)": (X_train_naive, y_train_naive, X_test_naive, y_test_naive, preprocess_naive, "naive"),
    "Preprocess (đầy đủ)": (X_train_full,  y_train_full,  X_test_full,  y_test_full,  preprocess_full,  "preprocess"),
}

for set_name, (Xtr, ytr, Xte, yte, prep, tag) in settings.items():
    for model_name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)
        cm_all = confusion_matrix(yte, y_pred, labels=[0, 1])

        fname = f"cm_{model_name.replace(' ', '')}_{tag}.png"
        fpath = out_dir / fname
        plot_cm_save(cm_all, f"{model_name} - {set_name}", fpath)
        print(f"  • {model_name:>18s} | {set_name} -> {fpath}")
