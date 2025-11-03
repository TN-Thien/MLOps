# Bài Thực Hành Buổi 1 — Ôn tập Machine Learning & FastAPI (Diabetes)

**Môn:** Machine Learning Operations
**Sinh viên:** *Đặng Công Thiên*
**Lớp** *23DAI*

## 1. Mục tiêu

1. Ôn tập quy trình đầy đủ của một bài toán học máy từ dữ liệu thô đến sản phẩm chạy được.
2. Làm quen với FastAPI để triển khai mô hình dưới dạng dịch vụ web.

## 2. Dữ liệu và biến sử dụng

* **Nguồn dữ liệu:** `diabetes.csv` (Pima Indians Diabetes).
* **Biến mục tiêu (target):** `Outcome` (0 = không mắc, 1 = mắc).
* **Đặc trưng đầu vào:**

  * **Số (numeric):** `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
  * **Phân loại (categorical):** `Race` (được mã hoá one-hot khi huấn luyện).

**Lý do xử lý các giá trị 0:** ở một số cột sinh học (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`), giá trị 0 là không hợp lý về mặt sinh lý nên được xem như thiếu dữ liệu; sau đó điền bằng **trung vị** để giảm thiên lệch. Cột `Outcome` giữ nguyên 0/1.

## 3. Quy trình thực hiện

### 3.1. Chuẩn bị dữ liệu

* Kiểm tra, chuyển các giá trị 0 bất thường → `NaN` ở các cột sinh học.
* Điền thiếu: **median** cho cột số; **giá trị phổ biến nhất** cho cột phân loại.
* Chuẩn hoá cột số bằng **StandardScaler**.
* Mã hoá `Race` bằng **One-Hot Encoder** (bỏ qua nhãn chưa gặp khi suy luận).
* Chia **train/test = 80/20** với **stratify** theo `Outcome` để giữ tỷ lệ lớp.

### 3.2. Xây dựng và huấn luyện mô hình

* Kiến trúc: **ColumnTransformer** (tiền xử lý số & phân loại) → **Logistic Regression** (`max_iter=1000`, `solver=liblinear`).
* Tối ưu tham số bằng **GridSearchCV (5-fold)** với lưới:

  * `C ∈ {0.01, 0.1, 1.0, 10.0}`
  * `class_weight ∈ {None, balanced}`
* Tiêu chí chọn mô hình: **ROC AUC** trung bình qua 5 fold.

### 3.3. Đánh giá và tối ưu

* Đánh giá trên tập test, ghi nhận các chỉ số tổng hợp và ma trận nhầm lẫn.
* Lưu **classification report** (precision/recall/F1 theo từng lớp) vào `report.csv`.
* Lưu mô hình đã huấn luyện thành **artifact** `diabetes_model.joblib`.

## 4. Kết quả đánh giá trên tập test

### 4.1. Chỉ số tổng hợp

* **Accuracy:** 0.7466
* **ROC AUC:** 0.8333
* **Average Precision (PR AUC):** 0.7458

Từ ma trận nhầm lẫn có thể suy ra các chỉ số của lớp dương (mắc bệnh):

* **Precision (lớp 1):** 37 / (37 + 24) ≈ 0.607
* **Recall (lớp 1):** 37 / (37 + 13) = 0.740
* **F1 (lớp 1):** ≈ 0.667
* **Specificity (lớp 0):** 72 / (72 + 24) = 0.750
* **Balanced accuracy:** (0.740 + 0.750) / 2 ≈ 0.745

### 4.2. Confusion Matrix và diễn giải

* **TN = 72:** Dự đoán đúng không mắc.
* **FP = 24:** Dự đoán nhầm là mắc trong khi thực tế không mắc.
* **FN = 13:** Dự đoán nhầm là không mắc trong khi thực tế mắc (bỏ sót).
* **TP = 37:** Dự đoán đúng mắc.

**Hình 1 — Confusion Matrix:**
`![Confusion Matrix - Logistic Regression](./confusion_matrix.png)`

## 5. Triển khai FastAPI

### 5.1. Thành phần

* **Endpoint `GET /`**: mô tả mô hình và liệt kê danh sách đặc trưng.
* **Endpoint `POST /predict`**: nhận JSON đầu vào, trả về `prediction` (0/1) và `probability_of_1`.
* **Lược đồ đầu vào** được kiểm tra/ép kiểu bằng **Pydantic BaseModel**.
* **Giao diện web `/ui`**: biểu mẫu HTML với thanh trượt cho từng tham số; hiển thị nguy cơ và phần trăm xác suất.

### 5.2. Hướng dẫn sử dụng (không trình bày mã)

1. Huấn luyện để sinh `diabetes_model.joblib` và `report.csv`.
2. Khởi động dịch vụ FastAPI.
3. Truy cập `http://127.0.0.1:8000/` để xem mô tả, `http://127.0.0.1:8000/docs` để thử nhanh, và `/ui` để dùng giao diện nhập.
4. Gửi yêu cầu `POST /predict` với các trường: `Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age, Race`. Dịch vụ trả kết quả dự đoán và xác suất ước lượng.

## 6. Kết luận

* Bài thực hành đã hoàn thành các yêu cầu: xây dựng mô hình, đánh giá, tối ưu đơn giản, xây dựng demo CLI và triển khai FastAPI (API + UI).
* Kết quả cho thấy mô hình đạt hiệu năng phù hợp với bộ dữ liệu; cách trình bày giúp sinh viên nắm được chu trình MLOps cơ bản từ dữ liệu tới dịch vụ.

## 7. Tệp nộp kèm theo

* `train.py` – huấn luyện & đánh giá; sinh `diabetes_model.joblib`, `report.csv` và (tuỳ chọn) `confusion_matrix.png`.
* `main.py` – dịch vụ FastAPI (`/`, `/predict`, `/ui`).
* `templates/index.html` – giao diện web.
* `demo_cli.py` – minh hoạ dự đoán qua dòng lệnh.
* `diabetes.csv` – dữ liệu.
* `README.md` – tài liệu này.

## 8. Hướng mở rộng

* Hiệu chuẩn xác suất (CalibratedClassifierCV), tối ưu ngưỡng theo mục tiêu (ví dụ Recall).
* Thử và so sánh thêm mô hình (RandomForest, XGBoost, SVM).
* Bổ sung đồ thị ROC/PR và Calibration Curve để minh hoạ.
* Đóng gói Docker để triển khai thực tế hoặc CI/CD cho quá trình huấn luyện–triển khai.
