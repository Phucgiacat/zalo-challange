import os
import json
import cv2
import time
import torch
import csv
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from transformers import AutoImageProcessor, AutoModel

MODEL_PATH = "/content/best.pt"
TEST_ROOT = "/content/public_test/samples"
OUTPUT_JSON = "jupyter_submission.json"
OUTPUT_CSV = "time_submission.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 896
CONF_THRESHOLD = 0.075
IOU_THRESHOLD = 0.7
MIN_DINO_SCORE = 0.025

class DroneDetector:
    def __init__(self):
        print(f"Initializing Detector on {DEVICE}...")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        self.yolo_model = YOLOWorld(MODEL_PATH)
        self.yolo_model.to(DEVICE)
        print(" - YOLO loaded.")
        try:
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE)
            print(" - DINOv2 loaded.")
        except Exception as e:
            print(f" - Error loading DINOv2: {e}")
            exit()

        self.current_ref_embs = []
        self.current_video_id = None

    def get_dino_embedding(self, pil_image):
        """Helper: Convert PIL image to DINO embedding."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten()

    def set_reference_images(self, video_id):
        """Pre-load reference embeddings for the current video context."""
        self.current_video_id = video_id
        self.current_ref_embs = []
        
        ref_dir = os.path.join(TEST_ROOT, video_id, "object_images")
        if not os.path.exists(ref_dir):
            return

        for img_name in ["img_1.jpg", "img_2.jpg", "img_3.jpg"]:
            img_path = os.path.join(ref_dir, img_name)
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    self.current_ref_embs.append(self.get_dino_embedding(pil_img))
                except Exception:
                    pass

    def predict_streaming(self, frame_bgr_np, frame_idx):
        results = self.yolo_model.predict(
            source=frame_bgr_np,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            device=DEVICE,
            max_det=300 
        )

        if not results or len(results[0].boxes) == 0:
            return None

        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = best_box.xyxy.cpu().numpy()[0]
        
        box_data = {
            "frame": frame_idx,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
        }

        if self.current_ref_embs:
            frame_rgb = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            w, h = pil_frame.size

            cx1, cy1, cx2, cy2 = max(0, box_data["x1"]), max(0, box_data["y1"]), min(w, box_data["x2"]), min(h, box_data["y2"])

            if (cx2 - cx1) > 5 and (cy2 - cy1) > 5:
                crop = pil_frame.crop((cx1, cy1, cx2, cy2))
                crop_emb = self.get_dino_embedding(crop)

                score = max([np.dot(crop_emb, ref) for ref in self.current_ref_embs])

                if score < MIN_DINO_SCORE:
                    return None 

        return box_data

def run_simulation():
    if not os.path.exists(TEST_ROOT):
        print(f"Error: Test folder not found at {TEST_ROOT}")
        return

    detector = DroneDetector()

    video_dirs = sorted([d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))])
    print(f"\nStarting simulation for {len(video_dirs)} videos...\n")

    final_json_data = []
    final_csv_data = []

    total_sim_time_ms = 0

    for i, vid in enumerate(video_dirs, 1):
        print(f"[{i}/{len(video_dirs)}] Processing {vid}...", end=" ", flush=True)

        detector.set_reference_images(vid)

        video_path = os.path.join(TEST_ROOT, vid, "drone_video.mp4")
        if not os.path.exists(video_path):
            print("Skipped (No video)")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        video_bboxes = []
        video_process_time_ms = 0.0

        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            t_start = time.time()
            result_box = detector.predict_streaming(frame_bgr, frame_idx)
            
            t_end = time.time()
            
            duration_ms = (t_end - t_start) * 1000
            video_process_time_ms += duration_ms

            if result_box:
                video_bboxes.append(result_box)

            frame_idx += 1

        cap.release()
        submission_id = f"jupyter_{vid}"
        detection_list = [{"bboxes": video_bboxes}] if video_bboxes else []
        final_json_data.append({
            "video_id": submission_id,
            "detections": detection_list
        })
        answer_str = json.dumps(detection_list)
        final_csv_data.append([submission_id, answer_str, int(video_process_time_ms)])

        print(f"Done | Frames: {frame_idx} | Found: {len(video_bboxes)} | Time: {int(video_process_time_ms)}ms")
        total_sim_time_ms += video_process_time_ms

    print("\n" + "="*50)
    print(f"Saving JSON submission to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_json_data, f, indent=2)
        
    print(f"Saving CSV submission to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'answer', 'time'])
        writer.writerows(final_csv_data)

    print(f"Total Simulation Time: {total_sim_time_ms:.0f} ms")
    print("="*50)

if __name__ == "__main__":
    run_simulation()