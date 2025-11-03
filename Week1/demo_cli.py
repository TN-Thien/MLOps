import json
import joblib
import pandas as pd

# 1. Tải mô hình đã lưu
MODEL_PATH = "diabetes_model.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
target_col = bundle["target_col"]

print("Mô hình đã tải từ:", MODEL_PATH)
print("Biến mục tiêu:", target_col)
print("Các đặc trưng đầu vào:", feature_cols)

# 2. Dữ liệu mẫu
sample_input = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
    "Race": "White"
}

print("Dữ liệu đầu vào mẫu:")
print(json.dumps(sample_input, ensure_ascii=False, indent=2))

# 3. Chuẩn bị dữ liệu
X = pd.DataFrame([sample_input], columns=feature_cols)

# 4. Dự đoán
if hasattr(model, "predict_proba"):
    proba = float(model.predict_proba(X)[0, 1])
else:
    proba = float(model.decision_function(X)[0])

pred = int(proba >= 0.5)

# 5. Kết quả
print("\nKết quả dự đoán:")
print("Nguy cơ mắc bệnh tiểu đường:", "Có thể mắc" if pred == 1 else "Ít khả năng")
print(f"Xác suất ước tính: {proba * 100:.2f}%")

# 6. Xuất kết quả dạng JSON
result = {
    "input": sample_input,
    "prediction": pred,
    "probability_of_1": proba
}
print("\nKết quả dạng JSON:")
print(json.dumps(result, ensure_ascii=False, indent=2))
