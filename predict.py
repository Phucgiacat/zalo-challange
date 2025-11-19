import os
import json
import cv2
import numpy  as np
import torch
from ultralytics import YOLOWorld


## --- Cấu hình ---
MODEl_PATH = "/code/saved_models/best.pt"
DATA_ROOT = "data/samples"
OUTPUT_JSON = "results/submission.json"

class AeroEyesModel:
    """
    Model Class with streaming prediction function
    """
    def __init__(self, model_path=MODEl_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        start_load_time = time.time()
        
        self.model = YOLOWorld(model_path)
        self.model.set_classes(["target"])
        
        self.model.to(self.device)
        self.model.eval()
        
        end_load_time = time.time()
        print(f"Model Load Time (ms): {int((end_load_time - start_load_time) * 1000)}")

    def predict_streaming(self, frame_rgb_np, frame_idx):
            """
            Dự đoán cho MỘT frame. 
            
            Args:
                frame_rgb_np: numpy array (H, W, 3) frame hiện tại (RGB)
                frame_idx: index của frame (tăng dần)
                
            Returns:
                list: [x1, y1, x2, y2] nếu có object, hoặc None 
            """
            # --- BẮT ĐẦU ĐO THỜI GIAN PREDICT TỪ FRAME IDX > 0 ---
            start_predict_time = time.time()
            
            results = self.model.predict(
                source=frame_rgb_np,
                conf=0.35,  
                iou=0.55,
                imgsz=896,
                verbose=False,
                device=self.device
            )
            
            boxes = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    box_xyxy = r.boxes.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    boxes.append([x1, y1, x2, y2])
            
            end_predict_time = time.time()
            
            print(f"Frame {frame_idx} Predict Time (ms): {int((end_predict_time - start_predict_time) * 1000)}")
            
            return boxes[0] if boxes else None

def main():
    """
    Hàm chính để mô phỏng quá trình chấm điểm bằng cách chạy inference trên tất cả videos.
    """
    model_instance = AeroEyesModel()

    video_dirs = sorted([d for d in os.listdir(DATA_ROOT) 
                         if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    submission = []
    
    for vid in video_dirs:
        video_path = os.path.join(DATA_ROOT, vid, "drone_video.mp4")
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        bboxes_per_frame = []
        frame_idx = 0
        
        print(f"Processing video: {vid}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            result_box = model_instance.predict_streaming(frame_rgb, frame_idx)

            if result_box is not None:
                x1, y1, x2, y2 = result_box
                bboxes_per_frame.append({
                    "frame": frame_idx,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
            
            frame_idx += 1
            
        cap.release()
        
        submission.append({
            "video_id": vid,
            "detections": [{"bboxes": bboxes_per_frame}] if bboxes_per_frame else []
        })
    if not os.path.exists(os.path.dirname(OUTPUT_JSON)):
        os.makedirs(os.path.dirname(OUTPUT_JSON))
        
    with open(OUTPUT_JSON, "w") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Finished processing all videos. Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()