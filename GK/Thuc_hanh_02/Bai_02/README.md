# Bài 2 – Tăng cường dữ liệu ảnh trên CIFAR-10

## 1. Giới thiệu

Trong bài thực hành này, em khảo sát ảnh hưởng của **tăng cường dữ liệu ảnh (Image Data Augmentation)** lên bài toán phân loại ảnh trên bộ dữ liệu CIFAR-10. Mục tiêu là so sánh hai kịch bản huấn luyện:

* Mô hình huấn luyện trên **dữ liệu gốc** (không augment).
* Mô hình huấn luyện trên **dữ liệu đã tăng cường**.

Việc so sánh dựa trên: độ chính xác trên tập kiểm tra, tốc độ hội tụ và mức độ overfitting.

---

## 2. Dữ liệu và tăng cường dữ liệu

### 2.1 Tập dữ liệu sử dụng

* Bộ dữ liệu: **CIFAR-10**, ảnh màu 32×32×3.
* Chọn 5 lớp: automobile, bird, frog, horse, ship.
* Tập huấn luyện: mỗi lớp lấy 1000 ảnh → tổng 5000 ảnh train.
* Tập kiểm tra: lấy toàn bộ ảnh thuộc 5 lớp trên, khoảng 5000 ảnh test.
* Nhãn được ánh xạ lại về 5 giá trị (0–4) tương ứng với 5 lớp.
* Tất cả ảnh được chuẩn hóa theo trung bình và độ lệch chuẩn của CIFAR-10 để ổn định quá trình huấn luyện.

### 2.2 Các phép tăng cường dữ liệu

Trong kịch bản dùng dữ liệu tăng cường, em áp dụng một chuỗi biến đổi ngẫu nhiên trên tập huấn luyện:

* **Lật ngang**: giúp mô hình không phụ thuộc hướng xuất hiện của đối tượng.
* **Xoay nhẹ**: mô phỏng các góc chụp khác nhau.
* **Dịch chuyển và phóng to/thu nhỏ**: cho phép đối tượng xuất hiện ở nhiều vị trí, kích thước khác nhau trong khung hình.
* **Cắt ngẫu nhiên rồi đưa về 32×32**: buộc mô hình chú ý đến nhiều vùng khác nhau của đối tượng.
* **Thay đổi độ sáng và độ tương phản**: mô phỏng điều kiện chiếu sáng đa dạng.

Đối với tập kiểm tra, em chỉ thực hiện các bước tiền xử lý cơ bản (chuyển định dạng, chuẩn hóa), không áp dụng các phép tăng cường để giữ nguyên phân phối dữ liệu gốc.

### 2.3 Minh hoạ ảnh gốc và ảnh sau tăng cường

Để minh hoạ, em chọn một ảnh từ mỗi lớp trong tập train. Với mỗi lớp, em trình bày:

* Ảnh gốc: chưa qua tăng cường.
* Ảnh sau tăng cường: đã qua chuỗi biến đổi ngẫu nhiên.

Các ảnh được sắp xếp thành một hình minh hoạ (Hình 1): hàng trên là ảnh gốc, hàng dưới là ảnh sau tăng cường, mỗi cột tương ứng một lớp. Quan sát cho thấy đối tượng có thể bị xoay nhẹ, dịch vị trí, cắt bớt hoặc thay đổi sáng/tối, nhưng vẫn nhận diện được. Điều này cho thấy data augmentation đã tạo thêm nhiều mẫu hợp lý về mặt ngữ nghĩa.

---

## 3. Mô hình và thiết kế thí nghiệm

### 3.1 Mô hình phân loại

Em sử dụng một mô hình **SmallResNet** gọn nhẹ, gồm các khối tích chập có kết nối dư (residual connection), phù hợp với ảnh kích thước 32×32. Phần cuối mô hình sử dụng cơ chế global average pooling và một lớp fully-connected để phân loại thành 5 lớp.

Mô hình được huấn luyện với:

* Hàm mất mát: CrossEntropyLoss.
* Thuật toán tối ưu: Adam với learning rate cố định và regularization L2 nhẹ.
* Batch size: 64.
* Số epoch: 10 cho mỗi lần chạy.

### 3.2 Kịch bản thí nghiệm

Em tiến hành hai kịch bản:

1. **Kịch bản 1 – Dữ liệu gốc**: tập train chỉ được chuẩn hóa, không áp dụng tăng cường.
2. **Kịch bản 2 – Dữ liệu tăng cường**: tập train được áp dụng đầy đủ các phép tăng cường đã nêu.

Mỗi kịch bản được chạy **3 lần** với các seed khác nhau. Sau khi huấn luyện, em thống kê trung bình và độ lệch chuẩn các chỉ số:

* Độ chính xác tốt nhất trên tập kiểm tra (best validation accuracy).
* Độ chính xác ở epoch cuối (final validation accuracy).
* Epoch đạt accuracy tốt nhất (tốc độ hội tụ).
* Thời gian huấn luyện toàn bộ.

Các chỉ số này được theo dõi và ghi lại bằng công cụ Weights & Biases (wandb).

---

## 4. Kết quả và nhận xét

### 4.1 Kết quả tổng hợp

Kết quả trung bình trên 3 lần chạy cho mỗi kịch bản:

* **Dữ liệu gốc (không augment)**:

  * Độ chính xác tốt nhất trên tập kiểm tra khoảng **82,34%**.
  * Epoch đạt kết quả tốt nhất nằm ở cuối quá trình huấn luyện.

* **Dữ liệu tăng cường**:

  * Độ chính xác tốt nhất trên tập kiểm tra khoảng **79,08%**.
  * Epoch đạt kết quả tốt nhất thường quanh cuối quá trình (xấp xỉ epoch 9–10).

Như vậy, trong thiết lập này, mô hình dùng dữ liệu gốc đạt accuracy cao hơn mô hình dùng dữ liệu tăng cường khoảng 3%.

### 4.2 Giải thích và thảo luận

Về lý thuyết, data augmentation giúp mở rộng không gian dữ liệu và giảm overfitting, từ đó thường cải thiện khả năng khái quát hóa. Tuy nhiên, trong thí nghiệm này, em quan sát thấy:

* Ở kịch bản không augment, độ chính xác trên tập train rất cao, trong khi độ chính xác trên tập test thấp hơn một chút. Điều này cho thấy mô hình có xu hướng overfitting.
* Ở kịch bản có augment, độ chính xác trên train thấp hơn nhưng lại gần với độ chính xác trên test hơn. Mô hình bớt overfitting nhưng đồng thời phải giải bài toán khó hơn do dữ liệu bị biến đổi mạnh.

Hai yếu tố quan trọng ảnh hưởng đến kết quả là:

1. **Số epoch huấn luyện còn hạn chế**: với chỉ 10 epoch, mô hình chưa có đủ thời gian để tận dụng hết lợi ích của dữ liệu đa dạng hơn.
2. **Mức độ tăng cường tương đối mạnh**: chuỗi biến đổi khiến phân phối dữ liệu train khác khá nhiều so với test, làm bài toán trở nên khó hơn trong ngắn hạn.

Do đó, việc accuracy của mô hình dùng augmentation thấp hơn trong thiết lập này không mâu thuẫn với lý thuyết, mà phản ánh sự đánh đổi giữa mức độ tăng cường, độ phức tạp mô hình và thời gian huấn luyện. Em kỳ vọng rằng nếu tăng số epoch hoặc điều chỉnh nhẹ độ mạnh của các phép augment, mô hình sử dụng dữ liệu tăng cường có thể đạt hoặc vượt kết quả của mô hình dùng dữ liệu gốc.

---

## 5. Kết luận

Qua bài thực hành, em đã:

* Xây dựng được một pipeline huấn luyện mô hình phân loại ảnh trên tập con CIFAR-10 với 5 lớp.
* Áp dụng nhiều kỹ thuật tăng cường dữ liệu và minh hoạ trực quan ảnh gốc so với ảnh sau tăng cường.
* Thực hiện thí nghiệm so sánh hai kịch bản (dữ liệu gốc và dữ liệu tăng cường) với nhiều lần chạy để có kết quả ổn định.

Kết quả cho thấy augmentation trong thiết lập hiện tại giúp giảm overfitting nhưng chưa giúp tăng accuracy trên tập kiểm tra. Điều này nhấn mạnh tầm quan trọng của việc lựa chọn mức độ tăng cường phù hợp và đủ thời gian huấn luyện khi áp dụng data augmentation trong thực tế.
