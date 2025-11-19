## **YOLO WORLD + MASKED TEMPORAL SELF-SUPERVISION CHO AEROEYES**

### 1. Ý tưởng Tổng Quan (Idea Overview)

Giải pháp của đội chúng tôi tập trung vào việc tận dụng tối đa dữ liệu video (temporal information) để tăng cường hiệu suất của mô hình Object Detection trong môi trường giám sát từ drone. [cite_start]Chúng tôi sử dụng kiến trúc **YOLO-World** (phiên bản `YOLOv8s-WorldV2` với khoảng **12.7 triệu tham số**) làm mô hình nền tảng, đảm bảo tuân thủ giới hạn 50M tham số.

Mục tiêu chính là huấn luyện mô hình không chỉ nhận diện vật thể trong từng frame mà còn học được **tính nhất quán và chuyển động** của vật thể theo thời gian, giúp cải thiện độ chính xác và độ bền (robustness) trong quá trình inference streaming.

### 2. Kỹ thuật Training (Training Strategy)

Chúng tôi áp dụng phương pháp **Masked Temporal Self-Supervised Learning** kết hợp với **Curriculum Learning** để chuẩn bị dữ liệu và huấn luyện mô hình:

#### 2.1. Masked Temporal Self-Supervised Learning
* **Mục đích:** Tạo ra các cặp dữ liệu (ảnh đầu vào, nhãn ground truth) giả lập, nơi mô hình phải học cách **dự đoán vị trí của vật thể trong các frame bị che (masked frames)** dựa trên các frame nhìn thấy được (visible frames) lân cận.
* **Thực hiện:**
    1.  [cite_start]**Masking:** Một tỷ lệ frames nhất định trong chuỗi video được chọn để "che" nhãn Ground Truth[cite: 474].
    2.  **Temporal Interpolation (Nội suy Temporal):** Vị trí Bounding Box (`x1`, `y1`, `x2`, `y2`) của vật thể trong các frame bị che được **nội suy** bằng phương pháp **Cubic Spline Interpolation** dựa trên các vị trí vật thể trong các frame nhìn thấy được. Kết quả nội suy này được sử dụng làm **Ground Truth** cho các frame bị che.
    3.  **Tác dụng:** Buộc mô hình học các tính năng Temporal và Tracking ngầm, giảm sự phụ thuộc vào các nhãn bị nhiễu (noisy labels) và tăng cường khả năng theo dõi.

#### 2.2. Curriculum Learning (CL)
Quá trình huấn luyện được chia thành 3 pha (Phase) để dần dần tăng độ khó của tác vụ Masking, giúp mô hình học từ dễ đến khó:

| Phase | Epochs (Reference) | Mask Ratio | Strategy | Độ Khó |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | 0-15 | 10% | **Random Mask** | Dễ nhất (chỉ che ngẫu nhiên) |
| **Phase 2** | 15-35 | 30% | **Span Mask** | Trung bình (che một đoạn liên tiếp) |
| **Phase 3** | 35-50 | 50% | **Keyframe Mask** | Khó nhất (ưu tiên che các frame có chuyển động lớn nhất) |

### 3. Kỹ thuật Inference (Inference Strategy)

Mô hình được tối ưu hóa cho tốc độ và tính nhất quán trên thiết bị drone (streaming environment):

* **Model:** Sử dụng file `best.pt` sau khi training (đã được copy vào `/code/saved_models/best.pt`).
* [cite_start]**Streaming API:** Logic dự đoán được đặt trong hàm `predict_streaming(frame_rgb_np, frame_idx)` của class `AeroEyesModel` trong file `predict.py`[cite: 301].
* **Output:** Hàm `predict_streaming` trả về `[x1, y1, x2, y2]` hoặc `None`.
* [cite_start]**Temporal Coherence:** Do mô hình đã được huấn luyện với dữ liệu Temporal-aware, nó có khả năng duy trì tính nhất quán của Box giữa các frame (frame $t-1, t-2, ...$ để dự đoán frame $t$)[cite: 306, 307].

### 4. Training Code và Reproducibility

* **Mã nguồn:** Toàn bộ logic Training (Data Augmentation, Curriculum, Model Initialization) được chứa trong file **`training_code/train.py`** và **`Self_Supervised.ipynb`**.
* **Model/Data Download:** Base model (`yolov8s-worldv2.pt`) và Data Observation/Test được tải xuống trong notebook. [cite_start]BTC có thể sử dụng các URL đã ghi rõ trong code/tài liệu nộp bài để tái tạo dữ liệu training.
* [cite_start]**Seed Cố Định:** **Seed** được cố định bằng **42** trong cả quá trình training (`train.py`) và inference (`predict_notebook.ipynb`), đảm bảo tính tái tạo (reproducibility) của kết quả.