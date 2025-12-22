# Project MLOps: Đánh giá chất lượng hình ảnh (IQA)

Project này xây dựng pipeline MLOps chạy hoàn toàn local cho bài toán Image Quality Assessment (IQA):
- Training (PyTorch) + đánh giá metrics
- MLflow Tracking + Model Registry (local)
- FastAPI Serving (API + UI upload ảnh + hiển thị kết quả trực quan)
- Docker Compose chạy toàn bộ stack (MLflow + API)
- CI (GitHub Actions): pytest + kiểm tra build Docker
- (Tuỳ chọn) CD local (Windows self-hosted runner): push lên GitHub là máy Windows tự redeploy bằng Docker Compose

Mục tiêu: demo vòng đời mô hình theo MLOps.

---

## 1. Yêu cầu trước khi chạy

1) Cài Python 3.11
2) Cài Git
3) Cài Docker Desktop (Windows) để chạy Compose
4) Dataset phải có:
- file CSV: data/koniq10k_distributions_sets.csv
- thư mục ảnh: data/512x384/ (chứa ảnh)

Nếu thiếu ảnh, train sẽ lỗi vì không đọc được file.

---

## 2. Cấu trúc thư mục chính (Project_CK)

Project_CK/
- data/512x384/ : ảnh dataset
- data/koniq10k_distributions_sets.csv : split/labels
- training/src/train.py : script train + MLflow log + registry + alias staging
- serving/api.py : FastAPI app + mount routers + mount static
- serving/templates/index.html : trang chủ UI
- serving/templates/predict.html : UI upload ảnh + preview + kết quả trực quan
- serving/static/style.css : CSS giao diện (có thể dùng background.webp)
- serving/utils/load_model.py : load model từ MLflow theo alias (staging)
- tests/ : pytest kiểm thử API
- docker-compose.yml : chạy MLflow + API local
- infra/Dockerfile.* : đóng gói docker cho MLflow và API
- infra/run_mlflow.bat : chạy MLflow local trên Windows (nếu không dùng docker)

---

## 3. Chạy theo cách đơn giản nhất (không Docker)

Bước 1: mở PowerShell tại thư mục Project_CK và tạo môi trường ảo
- Windows: cd Project_CK
- Windows: python -m venv .venv
- Windows: .\.venv\Scripts\activate

Bước 2: cài thư viện
- Windows: python -m pip install --upgrade pip
- Windows: pip install -r requirements.txt

Bước 3: chạy MLflow local
- Windows: infra\run_mlflow.bat
Mở MLflow UI: http://127.0.0.1:5000

Bước 4: train + đăng ký model vào MLflow Registry

Ví dụ EfficientNet:
- Windows: python -m training.src.train --epochs 3 --batch-size 16 --backbone efficientnet_b0 --alias staging --primary-metric srocc

Ví dụ MobileNet:
- Windows: python -m training.src.train --epochs 3 --batch-size 16 --backbone mobilenet_v2 --alias staging --primary-metric srocc

Bước 5: chạy FastAPI (API + UI)
- Windows: uvicorn serving.api:app --reload --host 127.0.0.1 --port 8000

Bước 6: mở UI và API docs
- Trang chủ: http://127.0.0.1:8000/
- UI dự đoán: http://127.0.0.1:8000/predict-ui
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## 4. Chạy bằng Docker Compose

Bước 1: mở PowerShell tại Project_CK
- Windows: cd Project_CK

Bước 2: build và chạy toàn bộ stack
- Windows: docker compose up -d --build

Bước 3: kiểm tra container
- Windows: docker compose ps

Bước 4: mở dịch vụ
- MLflow: http://127.0.0.1:5000
- API/UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- UI dự đoán: http://127.0.0.1:8000/predict-ui

Bước 5: dừng hệ thống
- Windows: docker compose down

Ghi chú: khi chạy bằng docker compose, FastAPI trong container sẽ kết nối MLflow qua hostname service (ví dụ http://mlflow:5000) theo cấu hình compose/env.

---

## 5. Cách dùng UI dự đoán

1) Mở http://127.0.0.1:8000/
2) Bấm “Mở giao diện upload ảnh” (đi tới /predict-ui)
3) Chọn ảnh, xem preview
4) Bấm “Chấm điểm”
5) Kết quả hiển thị trực quan (điểm 0–100 + thanh tiến độ + nhãn mức chất lượng)

API dự đoán:
- Endpoint: POST /predict
- Form-data field: file
- Response: {"quality": <0..100>}

---

## 6. Training metrics

Các metrics chính:
- srocc: Spearman rank correlation (quan trọng nhất với IQA vì đánh giá khả năng xếp hạng chất lượng ảnh)
- plcc: Pearson correlation (tương quan tuyến tính)
- mae/rmse: sai số trên thang 0–1
- mae_100/rmse_100: sai số quy đổi thang 0–100 để dễ đọc

Quy tắc chọn model “tốt”:
- Chọn model theo val_srocc cao nhất
- Nếu bằng nhau, chọn rmse_100 thấp hơn

Alias staging:
- staging là alias trỏ đến 1 version trong MLflow Registry
- FastAPI sẽ load model theo alias staging (tức staging chính là model đang dùng trên UI/serving)

Nếu staging đổi mà serving vẫn dùng model cũ:
- restart FastAPI hoặc triển khai reload cache (tuỳ cấu hình load_model)

---

## 7. Test (pytest)

Chạy test local:
- Windows: cd Project_CK
- Windows: pytest -q

Nếu thấy dạng “.... [100%]” nghĩa là tất cả test PASS.

---

## 8. CI (GitHub Actions) — chỉ cần 1 file CI-CD.yaml

GitHub chỉ chạy workflow khi file nằm ở đúng đường dẫn tính từ root repo:
- (repo root)/.github/workflows/CI-CD.yaml

Trường hợp repo đặt project trong thư mục Project_CK/ thì workflow phải chạy pytest trong Project_CK và set PYTHONPATH để import được “serving”.

Ví dụ workflow CI:
- Trigger khi push main hoặc pull_request main
- Cài dependencies
- pytest -q
- Build docker images để kiểm tra Dockerfile build được

Sau khi push, xem kết quả:
- GitHub repo -> Actions -> workflow “CI”
- màu xanh = PASS, màu đỏ = FAIL
