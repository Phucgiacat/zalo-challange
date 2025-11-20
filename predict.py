import os
import json
import cv2
import time
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOWorld
from transformers import AutoImageProcessor, AutoModel

# =========================================================
# ⚙️ CẤU HÌNH HỆ THỐNG & THAM SỐ
# =========================================================

MODEL_PATH = "/saved_models/models.pt"
DATA_ROOT = "/data/samples"
OUTPUT_JSON = "results/submission.json"

# --- Tham số YOLO ---
IMG_SIZE = 896          
CONF_THRESHOLD = 0.15   
IOU_THRESHOLD = 0.65    

# --- Tham số Trọng số & DINO ---
WEIGHT_YOLO = 0.7       
WEIGHT_DINO = 0.3       
FINAL_THRESHOLD = 0.25  

# =========================================================
# 🧠 CLASS MODEL (YOLO + DINO Integrated)
# =========================================================

class AeroEyesModel:
    """
    Class quản lý model, tích hợp YOLO-World detection và DINOv2 verification
    với cơ chế chấm điểm có trọng số.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Running on: {self.device}")
        
        start_load = time.time()
        
        # 1. Load YOLO-World
        print("⏳ Loading YOLO-World...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = YOLOWorld(model_path)
        # self.model.set_classes(["target"]) 
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Load DINOv2
        print("⏳ Loading DINOv2...")
        try:
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device)
        except Exception as e:
            print(f"❌ Error loading DINOv2: {e}")
            exit()
            
        # Biến lưu trữ embedding ảnh mẫu của video hiện tại
        self.current_ref_embs = []

        print(f"✅ Setup Complete. Load time: {(time.time() - start_load):.2f}s")

    def _get_dino_embedding(self, pil_image):
        """Chuyển ảnh PIL thành vector đặc trưng DINO"""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten()

    def prepare_for_video(self, video_id, data_root):
        """
        Load 3 ảnh mẫu (reference images) của video chuẩn bị xử lý.
        Hàm này phải được gọi trước khi vòng lặp frame bắt đầu.
        """
        self.current_ref_embs = []
        ref_dir = os.path.join(data_root, video_id, "object_images")
        
        if not os.path.exists(ref_dir):
            return

        for img_name in ["img_1.jpg", "img_2.jpg", "img_3.jpg"]:
            img_path = os.path.join(ref_dir, img_name)
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    self.current_ref_embs.append(self._get_dino_embedding(pil_img))
                except: pass

    def predict_streaming(self, frame_rgb_np, frame_idx):
        """
        Dự đoán cho MỘT frame đơn lẻ.
        Returns: [x1, y1, x2, y2] hoặc None
        """
        
        # --- BƯỚC 1: YOLO PREDICT ---
        results = self.model.predict(
            source=frame_rgb_np,
            conf=CONF_THRESHOLD,  
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            device=self.device
        )
        
        best_box = None
        max_conf = 0.0

        # Tìm box có confidence cao nhất trong frame (Top-1)
        for r in results:
            if r.boxes and len(r.boxes) > 0:
                confs = r.boxes.conf
                best_idx = torch.argmax(confs).item()
                
                current_conf = confs[best_idx].item()
                if current_conf > max_conf:
                    max_conf = current_conf
                    best_box = r.boxes[best_idx].xyxy.cpu().numpy()[0]

        # Nếu YOLO không thấy gì -> Bỏ qua
        if best_box is None:
            return None

        # --- BƯỚC 2: DINO VERIFICATION & WEIGHTING ---
        
        # Trường hợp hiếm: Không có ảnh mẫu -> Tin YOLO 100%
        if not self.current_ref_embs:
            if max_conf > FINAL_THRESHOLD:
                return [int(x) for x in best_box]
            return None

        # Lấy tọa độ và fix biên
        x1, y1, x2, y2 = map(int, best_box)
        h_img, w_img, _ = frame_rgb_np.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # ⚠️ QUAN TRỌNG: Loại bỏ box quá nhỏ để tránh lỗi "Ambiguous dimension"
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        # Crop ảnh & Lấy Embedding
        crop_np = frame_rgb_np[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop_np)
        crop_emb = self._get_dino_embedding(pil_crop)
        
        # Tính độ giống nhau (Similarity)
        dino_score = max([np.dot(crop_emb, ref) for ref in self.current_ref_embs])

        # 🔥 TÍNH ĐIỂM TỔNG HỢP (WEIGHTED SCORE)
        combined_score = (max_conf * WEIGHT_YOLO) + (dino_score * WEIGHT_DINO)

        # Debug: In ra để căn chỉnh ngưỡng (nếu cần)
        # print(f"F{frame_idx}: YOLO={max_conf:.2f} | DINO={dino_score:.2f} | Final={combined_score:.2f}")

        if combined_score > FINAL_THRESHOLD:
            return [x1, y1, x2, y2]
        
        return None

# =========================================================
# 🚀 MAIN EXECUTION FLOW
# =========================================================

def main():
    # 1. Khởi tạo Model
    try:
        model_instance = AeroEyesModel()
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    if not os.path.exists(DATA_ROOT):
        print(f"❌ Error: Data root not found at {DATA_ROOT}")
        return

    video_dirs = sorted([d for d in os.listdir(DATA_ROOT) 
                         if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    submission = []
    total_process_time = 0
    
    print(f"\n🚀 Start processing {len(video_dirs)} videos...\n")

    # 2. Loop qua từng video
    for vid_idx, vid in enumerate(video_dirs, 1):
        video_path = os.path.join(DATA_ROOT, vid, "drone_video.mp4")
        if not os.path.exists(video_path):
            print(f"⚠️ Skip {vid} (Video not found)")
            continue

        print(f"[{vid_idx}/{len(video_dirs)}] Processing {vid}...", end=" ", flush=True)

        # --- A. Chuẩn bị ảnh mẫu cho video này ---
        model_instance.prepare_for_video(vid, DATA_ROOT)

        # --- B. Mở video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        bboxes_per_frame = []
        frame_idx = 0
        
        # --- START TIMER (Per Video) ---
        start_vid_time = time.time()
        
        # --- C. Loop từng frame ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Chuyển BGR -> RGB (Quan trọng cho cả YOLO và DINO)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Gọi hàm dự đoán
            result_box = model_instance.predict_streaming(frame_rgb, frame_idx)

            if result_box is not None:
                x1, y1, x2, y2 = result_box
                bboxes_per_frame.append({
                    "frame": frame_idx,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
            
            frame_idx += 1
            
        cap.release()
        
        # --- END TIMER ---
        duration = time.time() - start_vid_time
        total_process_time += duration
        
        print(f"✅ Done in {duration:.2f}s | Found: {len(bboxes_per_frame)} frames")
        
        # --- D. Lưu kết quả vào list ---
        submission.append({
            "video_id": vid,
            "detections": [{"bboxes": bboxes_per_frame}] if bboxes_per_frame else []
        })

    # 3. Xuất file JSON cuối cùng
    out_dir = os.path.dirname(OUTPUT_JSON)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("\n" + "="*50)
    print(f"💾 Saving results to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"⏱️ Total Execution Time: {total_process_time:.2f}s")
    if len(video_dirs) > 0:
        print(f"⏱️ Average Time per Video: {total_process_time/len(video_dirs):.2f}s")
    print("="*50)

# if _name_ == "_main_":
#     main()