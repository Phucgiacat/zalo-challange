# YOLO-World Enhanced Pipeline: Video Object Detection

Dự án này cung cấp một pipeline huấn luyện mô hình YOLO nâng cao dành cho bài toán phát hiện vật thể trong video (Video Object Detection). Code không chỉ sử dụng phương pháp huấn luyện tiêu chuẩn mà còn tích hợp các kỹ thuật tiên tiến như **Self-Supervised Masking**, **Curriculum Learning**, và **Hard Negative Mining** để tăng cường độ chính xác và tính bền vững của mô hình.

## 📋 Mục Lục

1.  Cài đặt & Yêu cầu
2.  Hướng dẫn chạy
3.  Cấu hình & Tham số
4.  Ý tưởng thực hiện (Methodology)
5.  Điểm mạnh & Hạn chế
-----

## 🛠 Cài đặt & Yêu cầu

Dự án được thiết kế để chạy trên môi trường Google Colab hoặc Local với hỗ trợ GPU.

### Thư viện chính

  * Python 3.8+
  * Torch, Torchvision
  * Ultralytics (YOLO)
  * OpenCV (cv2)
  * Supervision
  * Scipy, Numpy, PyYAML

### Cài đặt

Chạy cell đầu tiên trong notebook để cài đặt các dependencies cần thiết:

```bash
pip install ultralytics supervision torch torchvision transformers opencv-python
```

-----

## 🚀 Hướng dẫn chạy

### Bước 1: Chuẩn bị dữ liệu

Trong file notebook (`main.ipynb`), Cell số 2 sử dụng `gdown` để tải dữ liệu từ Google Drive.

  * Nếu chạy trên Colab: Giữ nguyên code để tải và giải nén tự động.
  * Nếu chạy Local: Hãy đảm bảo bạn đã tải file `observing.zip` và `public_test.zip` vào đúng thư mục `/content/Data/`.

### Bước 2: Chạy Pipeline

Thực thi lần lượt các cell trong Notebook. Quy trình chính nằm ở hàm `main_pipeline()` bao gồm:

1.  Load config và chia tập Train/Val.
2.  **Data Generation:** Trích xuất frame từ video, áp dụng Masking và Hard Negative Mining.
3.  Tạo file `data.yaml`.
4.  **Training:** Huấn luyện model YOLO với chiến lược Curriculum Learning.
5.  **Analysis:** Kiểm tra số lượng tham số (Parameter Count).

### Bước 3: Inference (Dự đoán)

Sau khi train xong, code sẽ tự động gọi hàm `run_inference()` để chạy trên tập `public_test` và xuất ra file `submission_optimized_v2.json`.

-----

## ⚙️ Cấu hình & Tham số

Mọi cấu hình quan trọng đều nằm trong class `Config` (Cell 4) và `main_pipeline` (Cell 8).

### 1\. Cấu hình Model & Dữ liệu (Class `Config`)

| Tham số | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `MODEL_WEIGHTS` | `"yolo12s.pt"` | File weight khởi tạo (Pre-trained). Có thể đổi thành `yolov8s.pt`, `yolo11s.pt`. |
| `PARAM_LIMIT` | `50_000_000` | Giới hạn tham số tối đa (dùng để check luật cuộc thi). |
| `WORK_DIR` | `"enhanced_mixed_dataset_v2"` | Thư mục chứa ảnh/label đã xử lý. |
| `TRAIN_RATIO` | `0.8` | Tỉ lệ chia tập Train (80%) và Validation (20%). |

### 2\. Cấu hình Kỹ thuật nâng cao

| Tham số | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `ENABLE_MASKING` | `True` | Bật/Tắt tính năng che khung hình (Masking). |
| `BACKGROUND_FRAME_RATIO`| `0.1` | Tỉ lệ thêm frame nền (không có vật thể) để model học **Hard Negative** (giảm báo sai). |
| `CURRICULUM` | (Dict) | Cấu hình lộ trình học: Giai đoạn 1 (Dễ - Mask ít), Giai đoạn 3 (Khó - Mask nhiều). |

### 3\. Cấu hình Training (`model.train` trong `main_pipeline`)

  * `epochs`: Số vòng lặp huấn luyện (Mặc định: 15).
  * `imgsz`: Kích thước ảnh đầu vào (Mặc định: 896). Giảm xuống 640 nếu GPU yếu.
  * `freeze`: Số lớp bị đóng băng (không train lại backbone) để giữ kiến thức gốc.
  * `lr0`, `lrf`: Learning rate khởi tạo và hệ số giảm.

-----

## 💡 Ý tưởng thực hiện (Methodology)

Pipeline này giải quyết bài toán phát hiện vật thể trong video bằng cách kết hợp 3 chiến lược mũi nhọn:

### 1\. Self-Supervised Masking Strategy

Thay vì chỉ đưa ảnh tĩnh vào mô hình, code sử dụng thuật toán nội suy (`interpolate_boxes`) để tạo ra Ground Truth cho các khung hình bị che (masked).

  * **Cách hoạt động:** Ngẫu nhiên che đi một số frame hoặc che theo chuỗi (span).
  * **Mục đích:** Buộc mô hình phải học cách "đoán" vị trí vật thể dựa trên ngữ cảnh temporal (thời gian), giúp mô hình bền vững hơn khi vật thể bị che khuất hoặc mờ trong thực tế.

### 2\. Curriculum Learning (Học theo lộ trình)

Không ném dữ liệu khó vào ngay từ đầu. Class `CurriculumController` chia quá trình chuẩn bị dữ liệu thành 3 pha:

  * **Phase 1:** Masking tỉ lệ thấp (Dễ).
  * **Phase 2 & 3:** Tăng dần tỉ lệ Masking (Khó dần).
  * **Tác dụng:** Giúp mô hình hội tụ nhanh hơn và tránh bị "sốc" dữ liệu nhiễu ở những epoch đầu.

### 3\. Hard Negative Mining (HNM)

Code chủ động trích xuất các frame **không có vật thể** (background frames) và đưa vào tập train với label rỗng.

  * **Tác dụng:** Dạy cho mô hình biết "đây là nền, không phải vật thể", giúp giảm đáng kể tỉ lệ **False Positive** (báo giả).

-----

## 📊 Điểm mạnh & Hạn chế

### ✅ Điểm mạnh

1.  **Tính bền vững cao:** Nhờ Masking Strategy, model có khả năng nhận diện tốt hơn trong điều kiện video bị rung lắc, vật thể bị che khuất một phần.
2.  **Giảm báo sai (False Positives):** Kỹ thuật HNM cực kỳ hiệu quả trong việc loại bỏ các hộp bounding box rác ở background.
3.  **Tối ưu hóa tài nguyên:** Có tích hợp `ModelOptimizer` để đếm tham số và Pruning (cắt tỉa) nếu model vượt quá giới hạn cho phép.
4.  **Pipeline tự động hóa:** Từ khâu tải data, xử lý ảnh, nội suy nhãn đến training và inference đều được code liền mạch.

### ❌ Hạn chế

1.  **Thời gian tiền xử lý lâu:** Việc trích xuất frame từ video và chạy thuật toán nội suy (Interpolation) tốn nhiều thời gian CPU và ổ cứng hơn so với cách train truyền thống.
2.  **Phụ thuộc vào Interpolation:** Nhãn (Label) sinh ra từ nội suy là nhãn giả định (pseudo-label). Nếu vật thể chuyển động phi tuyến tính quá phức tạp, nhãn này có thể bị lệch so với thực tế.
3.  **Dung lượng lưu trữ:** Việc bung frame ra file ảnh (`.jpg`) sẽ tốn dung lượng đĩa đáng kể so với việc đọc trực tiếp từ video (tuy nhiên cách này giúp YOLO train nhanh hơn).

-----

### 👨‍💻 Tác giả
1. [Dương Hoài Minh]()
2. [Tống Trọng Tâm]()
3. [Bùi Hồng Phúc]()