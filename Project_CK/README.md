# Project MLOps: Đánh giá chất lượng hình ảnh

Dự án này xây dựng một bài toán đánh giá chất lượng hình ảnh, bao gồm toàn bộ vòng đời của mô hình học máy:

* Chuẩn bị dữ liệu
* Huấn luyện mô hình (PyTorch)
* Đánh giá bằng metric chuẩn IQA (Image Quality Assessment)
* Quản lý experiment & model bằng MLflow Tracking + Model Registry
* Triển khai mô hình bằng FastAPI (API + Web UI)
* Kiểm thử tự động với pytest
* Đóng gói & vận hành bằng Docker / Docker Compose

Mục tiêu của project:

> Minh họa một quy trình MLOps thực tế, có thể mở rộng, dễ demo và dễ deploy, thay vì chỉ train model đơn lẻ.


## 1. Bài toán: Đánh giá chất lượng hình ảnh

Đánh giá chất lượng hình ảnh là bài toán ước lượng mức độ “tốt/xấu” của một ảnh dựa trên cảm nhận thị giác của con người.

* Input: 1 ảnh RGB
* Output: 1 điểm số 0 – 100 (điểm càng cao → ảnh càng đẹp)

Trong project này:

* Mô hình được huấn luyện để dự đoán điểm chuẩn hoá 0 – 1
* Khi serving, điểm được quy đổi sang thang 0 – 100 để dễ hiểu với người dùng


## 2. Dataset
Project sử dụng dataset KonIQ-10k (Konstanz Natural Image Quality) – một dataset IQA phổ biến, được xây dựng dựa trên đánh giá chủ quan của con người với cấu trúc:

* Ảnh: `data/512x384/<image_name>`
* Nhãn & split: `data/koniq10k_distributions_sets.csv`

Nguồn dataset (Kaggle): https://www.kaggle.com/datasets/generalhawking/koniq-10k-dataset

### 2.1 Phân chia dữ liệu

| Split      | Số lượng |
| ---------- | -------- |
| Training   | 7,058    |
| Validation | 1,000    |
| Test       | 2,015    |


## 3. Mô hình học máy

Bài toán được giải dưới dạng regression với output là 1 giá trị liên tục.

### 3.1 Các backbone được sử dụng

Project so sánh 3 backbone CNN, đại diện cho các trade-off khác nhau:

| Backbone        | Đặc điểm chính                             |
| --------------- | ------------------------------------------ |
| EfficientNet-B0 | Hiện đại, hiệu quả, chất lượng dự đoán cao |
| ResNet18        | Ổn định, dễ huấn luyện, baseline tốt       |
| MobileNetV2     | Nhẹ, nhanh, phù hợp deploy                 |

Tất cả backbone đều:

* Load pretrained weights từ ImageNet
* Thay head cuối bằng tầng hồi quy 1 output
* Dùng Sigmoid để đảm bảo output nằm trong `[0, 1]`


## 4. Huấn luyện & đánh giá

### 4.1 Metric sử dụng

Do IQA là bài toán đánh giá thứ hạng chất lượng ảnh, project sử dụng các metric chuẩn:

* SROCC (Spearman Rank Correlation) – *metric chính*
* PLCC (Pearson Linear Correlation)
* MAE, RMSE
* MAE_100, RMSE_100 (quy đổi về thang 0–100)

> Model được chọn dựa trên SROCC validation cao nhất. Nếu bằng nhau, so sánh RMSE_100.


### 4.2 Cấu hình huấn luyện

Mỗi backbone được huấn luyện với 3 cấu hình hyperparameter:

| Learning rate | Epoch | Mục đích                                |
| ------------- | ----- | --------------------------------------- |
| 1e-3          | 3     | Học nhanh, kiểm tra khả năng hội tụ sớm |
| 3e-4          | 5     | Cân bằng giữa tốc độ và độ ổn định      |
| 1e-4          | 6     | Fine-tuning chậm, ổn định, giảm overfit |

→ Tổng cộng: 9 experiment (3 model × 3 cấu hình)


## 5. MLflow – Tracking & Model Registry

Project sử dụng MLflow local cho toàn bộ vòng đời mô hình:

* Log hyperparameter, metric theo từng epoch
* So sánh experiment trực quan
* Log model kèm input_example & signature
* Đăng ký model vào Model Registry

### 5.1 Alias `staging`

* Alias `staging` luôn trỏ tới model tốt nhất hiện tại
* Khi train model mới:

  * Nếu model tốt hơn (theo SROCC) → tự động cập nhật alias
  * Nếu không → giữ nguyên model đang dùng

FastAPI serving luôn load model theo alias, không phụ thuộc version cụ thể.


## 6. Serving – FastAPI (API + Web UI)

### 6.1 API

Các endpoint chính:

* `GET /health` – kiểm tra server
* `POST /predict` – upload ảnh, trả về điểm chất lượng
* `POST /reload-model` – reload model từ MLflow (clear cache)
* `GET /docs` – Swagger UI

### 6.2 Web UI

Project cung cấp giao diện web đơn giản:

* Upload ảnh
* Preview ảnh
* Hiển thị điểm số (0–100)
* Thanh progress bar trực quan
* Nhãn đánh giá: *Kém / Trung bình / Tốt / Rất tốt*

UI giúp demo nhanh và trực quan cho người không chuyên ML.


## 7. Kiểm thử (Testing)

Project sử dụng pytest để kiểm thử:

* Health endpoint
* Root endpoint (`/`)
* Predict endpoint

Trong test predict:

* Model được mock để tránh phụ thuộc MLflow
* Đảm bảo test chạy nhanh, ổn định

Chạy test:

```bash
pytest -q
```


## 8. Docker & Docker Compose

Project hỗ trợ chạy toàn bộ stack bằng Docker:

* MLflow server
* FastAPI serving

Chạy hệ thống:

```bash
docker compose up -d --build
```

Dừng hệ thống:

```bash
docker compose down
```


## 9. Cách chạy nhanh (không Docker)

1. Tạo môi trường ảo và cài thư viện

```bash
pip install -r requirements.txt
```

2. Chạy MLflow

```bash
infra/run_mlflow.bat
```

3. Huấn luyện model

```bash
python -m training.src.train --backbone efficientnet_b0 --epochs 5 --lr 3e-4 --alias staging
```

4. Chạy FastAPI

```bash
uvicorn serving.api:app --reload
```

5. Mở trình duyệt:

* Trang chủ: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* UI dự đoán: [http://127.0.0.1:8000/predict-ui](http://127.0.0.1:8000/predict-ui)


## 10. Tổng kết

Project này thể hiện:

* Một pipeline MLOps end-to-end
* So sánh nhiều mô hình & hyperparameter một cách có hệ thống
* Tách biệt rõ training – model registry – serving
* Có thể dùng làm:

  * Project portfolio
  * Demo phỏng vấn
  * Nền tảng mở rộng production
