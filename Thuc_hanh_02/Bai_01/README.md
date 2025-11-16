# Thực hành 02 – Phân loại thu nhập (UCI Adult)

## Tóm tắt

Báo cáo này trình bày về bài toán phân loại nhị phân thu nhập cá nhân trên tập dữ liệu Adult (UCI), với mục tiêu dự đoán hai lớp `>50K` và `<=50K`. Em so sánh hiệu năng mô hình trong hai trường hợp: (i) chưa áp dụng tiền xử lý đầy đủ (Naive) và (ii) áp dụng tiền xử lý đầy đủ (Preprocess). Ba mô hình được khảo sát gồm Logistic Regression, Random Forest và Gradient Boosting. Hiệu năng được đánh giá bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix. Kết quả cho thấy tiền xử lý đầy đủ cải thiện rõ rệt Logistic Regression, trong khi tác động lên Random Forest và Gradient Boosting là nhỏ hoặc cận biên. Gradient Boosting trong trường hợp Preprocess đạt kết quả tốt nhất tổng thể (F1 cao nhất).

---

## 1. Dữ liệu và bài toán

Adult (UCI) là tập dữ liệu chuẩn cho bài toán dự đoán thu nhập hằng năm của cá nhân vượt ngưỡng 50.000 USD. Mỗi mẫu là một cá nhân với tập thuộc tính hỗn hợp số và hạng mục (tuổi, trình độ học vấn, nghề nghiệp, số giờ làm việc/tuần, v.v.). Nhãn mục tiêu gồm hai giá trị: `>50K` và `<=50K`.

* Chuẩn hoá nhãn tập kiểm tra: nhãn trong tệp kiểm tra có ký tự dấu chấm ở cuối; chúng tôi loại bỏ để thống nhất định dạng.
* Biến số: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
* Biến hạng mục: tất cả biến còn lại trừ biến mục tiêu `income`.

---

## 2. Tiền xử lý

### 2.1. Trường hợp Naive

* Giá trị thiếu biểu diễn bằng `?` được chuyển thành thiếu và loại bỏ toàn bộ hàng chứa thiếu.
* Biến hạng mục được mã hoá thứ bậc (OrdinalEncoder) để đưa vào mô hình; không chuẩn hoá các biến số.

### 2.2. Trường hợp Preprocess (đầy đủ)

* Xử lý thiếu: biến số điền giá trị trung vị (median); biến hạng mục điền giá trị phổ biến (most_frequent).
* Mã hoá hạng mục: One-Hot Encoding với cơ chế bỏ qua giá trị chưa thấy (`handle_unknown=ignore`).
* Chuẩn hoá biến số: StandardScaler để đưa các thang đo về cùng phân phối (trung bình 0, độ lệch chuẩn 1).

> Trường hợp Naive làm giảm cỡ mẫu do loại bỏ hàng thiếu; Preprocess giữ lại thông tin thông qua cơ chế điền thiếu.

---

## 3. Mô hình và cấu hình

Ba mô hình tiêu biểu ở ba thuật toán khác nhau:

* Logistic Regression (LR) – mô hình tuyến tính tối ưu hoá log-loss; sử dụng `class_weight=balanced` để giảm lệch lớp; số vòng lặp tối đa vừa đủ để hội tụ.
* Random Forest (RF) – tổ hợp nhiều cây quyết định độc lập, trung bình hoá dự đoán để giảm phương sai.
* Gradient Boosting (GB) – mô hình hoá tăng cường theo giai đoạn, từng bước học trên phần sai số còn lại, cho khả năng biểu diễn phi tuyến tốt.

---

## 4. Thiết kế thực nghiệm

* Hai trường hợp: Naive (loại thiếu, OrdinalEncoder, không chuẩn hoá) và Preprocess (impute + One-Hot + chuẩn hoá).
* Bộ kiểm tra: sử dụng tập kiểm tra chuẩn của Adult; mọi phép biến đổi được học trên huấn luyện và áp dụng lên kiểm tra qua Pipeline.
* Chỉ số đánh giá:

  * Accuracy: tỉ lệ dự đoán đúng trên toàn bộ mẫu.
  * Precision (lớp `>50K`): tỉ lệ mẫu dự đoán dương tính thật trên tổng dự đoán dương tính.
  * Recall (lớp `>50K`): tỉ lệ mẫu dương tính thật được phát hiện trên tổng dương tính thật.
  * F1-score: trung bình điều hoà giữa Precision và Recall.
  * Confusion Matrix: đếm TN, FP, FN, TP theo thứ tự nhãn `<=50K` (0) và `>50K` (1).

> Việc nhấn mạnh F1 và Recall là phù hợp trong bối cảnh lớp `>50K` thường ít hơn, giúp đánh giá cân bằng giữa độ chính xác và khả năng bao phủ lớp dương tính.

---

## 5. Kết quả thực nghiệm

### 5.1. Bảng tổng hợp theo từng trường hợp

| Trường hợp | Mô hình            | Accuracy | Precision | Recall |     F1 |    TN |   FP |   FN |   TP |
| ---------- | ------------------ | -------: | --------: | -----: | -----: | ----: | ---: | ---: | ---: |
| Naive      | LogisticRegression |   0.7310 |    0.4696 | 0.7330 | 0.5725 |  8297 | 3063 |  988 | 2712 |
| Naive      | RandomForest       |   0.8513 |    0.7389 | 0.6103 | 0.6684 | 10562 |  798 | 1442 | 2258 |
| Naive      | GradientBoosting   |   0.8629 |    0.7882 | 0.6046 | 0.6843 | 10759 |  601 | 1463 | 2237 |
| Preprocess | LogisticRegression |   0.8063 |    0.5600 | 0.8404 | 0.6721 |  9896 | 2539 |  614 | 3232 |
| Preprocess | RandomForest       |   0.8528 |    0.7278 | 0.6022 | 0.6591 | 11569 |  866 | 1530 | 2316 |
| Preprocess | GradientBoosting   |   0.8689 |    0.7920 | 0.6037 | 0.6852 | 11825 |  610 | 1524 | 2322 |

### 5.2. Ảnh hưởng của tiền xử lý (Preprocess − Naive)

| Mô hình            | ΔAccuracy | ΔPrecision | ΔRecall |     ΔF1 |
| ------------------ | --------: | ---------: | ------: | ------: |
| LogisticRegression |   +0.0753 |    +0.0904 | +0.1074 | +0.0997 |
| RandomForest       |   +0.0016 |    −0.0110 | −0.0081 | −0.0094 |
| GradientBoosting   |   +0.0060 |    +0.0037 | −0.0009 | +0.0009 |

## 5.3. Hình ảnh Confusion Matrix

Các hình minh hoạ Confusion Matrix cho từng mô hình và từng trường hợp (Naive, Preprocess) được lưu trong thư mục [`outputs`](./outputs/). Các biểu đồ này giúp quan sát trực quan phân bố TN, FP, FN, TP, hỗ trợ so sánh mức độ bỏ sót và nhầm lẫn giữa các mô hình. Việc đối chiếu bảng số liệu với hình ảnh trực quan giúp đánh giá toàn diện hơn, đặc biệt trong các mô hình có chênh lệch Precision–Recall rõ rệt.

---

## 6. Phân tích kết quả

### 6.1. Tác động của tiền xử lý

* Logistic Regression được lợi rõ rệt từ Preprocess: One-Hot chuyển không gian hạng mục thành cơ sở đặc trưng tuyến tính; StandardScaler đưa các biến số về cùng thang đo, giúp tối ưu hoá ổn định; imputation bảo toàn cỡ mẫu. Hệ quả là F1 tăng khoảng 0.10, Recall và Precision đều cải thiện đáng kể.
* Random Forest gần như không cải thiện hoặc giảm nhẹ F1. Các mô hình dựa trên cây ít nhạy với chuẩn hoá; tăng số chiều do One-Hot chưa đem lại lợi ích với cấu hình hiện tại.
* Gradient Boosting cải thiện nhỏ (F1 +0.0009). Bản chất boosting đã mô hình hoá phi tuyến hiệu quả trên dữ liệu gốc; tiền xử lý chủ yếu tăng tính ổn định.

### 6.2. Mô hình tốt nhất và diễn giải

* Mô hình có kết quả tốt nhất là Gradient Boosting (Preprocess) với Accuracy 0.8689, F1 0.6852 – cao nhất tổng thể.
* Confusion Matrix (GB – Preprocess): TN = 11825, FP = 610, FN = 1524, TP = 2322. Precision xấp xỉ 0.792 (ít dương tính giả), Recall khoảng 0.604 (còn bỏ sót một phần dương tính thật). Tuỳ mục tiêu nghiệp vụ, có thể ưu tiên tăng Recall (giảm ngưỡng quyết định, cân bằng lớp, tuning siêu tham số) để phát hiện nhiều hơn các cá nhân có khả năng thu nhập cao.

---

## 7. Hạn chế và hướng khắc phục

* Mất cân bằng lớp: lớp `>50K` ít hơn có thể làm lệch mô hình. Có thể sử dụng `class_weight` và xem xét điều chỉnh ngưỡng hoặc áp dụng kỹ thuật tái lấy mẫu (ví dụ SMOTE).
* Độ nhạy với tiền xử lý: Logistic Regression phụ thuộc mạnh vào mã hoá/chuẩn hoá; Random Forest và Gradient Boosting ít nhạy hơn. Kết quả củng cố tính phù hợp của One-Hot kết hợp chuẩn hoá cho mô hình tuyến tính.
* Các hạn chế khác: chưa tối ưu siêu tham số có hệ thống; chưa phân tích lỗi chi tiết theo nhóm nhân khẩu học (nguy cơ thiên lệch). Những hạn chế này có thể ảnh hưởng đến mức độ tin cậy của kết luận và khả năng áp dụng kết quả cho các điều kiện thực tế khác.

---

## 8. Kết luận và hướng phát triển

Trong bài thực hành này, em so sánh ba mô hình Logistic Regression, Random Forest và Gradient Boosting cho bài toán phân loại thu nhập trên tập Adult, dưới hai thiết lập tiền xử lý: Naive và Preprocess. Kết quả cho thấy tiền xử lý đầy đủ có tác động khác nhau tuỳ mô hình: Logistic Regression được cải thiện rõ rệt nhờ imputation, One-Hot Encoding và chuẩn hoá; trong khi Random Forest và Gradient Boosting chỉ thay đổi nhẹ về hiệu năng.

Ở cấu hình hiện tại, Gradient Boosting với pipeline Preprocess là mô hình có hiệu năng tổng thể tốt nhất (Accuracy khoảng 0.87, F1 khoảng 0.69), ưu tiên Precision của lớp `>50K` nhưng vẫn giữ mức Recall chấp nhận được. Tuỳ theo mục tiêu ứng dụng (ưu tiên phát hiện đúng hay phủ hết các trường hợp thu nhập cao), có thể điều chỉnh ngưỡng quyết định hoặc áp dụng các kỹ thuật cân bằng lớp để dịch chuyển điểm cân bằng giữa Precision và Recall.

Các bước mở rộng hợp lý trong tương lai gồm: tối ưu siêu tham số cho từng mô hình; đánh giá chéo k-fold và bổ sung các thước đo như ROC-AUC, PR-AUC; phân tích lỗi chi tiết theo nhóm nhân khẩu học để phát hiện và giảm thiểu thiên lệch; và thử nghiệm thêm các đặc trưng biến đổi, chẳng hạn log-transform cho `capital-gain`/`capital-loss` hoặc tương tác giữa giờ làm việc và nghề nghiệp.