# ============================================
# train.py
# Nằm trong thư mục training_code/
# Chứa toàn bộ logic Data Preparation và Training Model
# ============================================

import os
import json
import cv2
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLOWorld

# ============================================
# I. CONFIGURATION (Lấy từ notebook)
# ============================================

class Config:
    """Cấu hình cho pipeline training"""
    # Dataset paths
    # Giả sử thư mục 'train' (data gốc) nằm cùng cấp với 'training_code' 
    # trong folder gốc của dự án khi BTC reproduce
    DATASET_ROOT = "../train" 
    ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, "annotations/annotations.json")
    SAMPLES_DIR = os.path.join(DATASET_ROOT, "samples")
    
    # OUTPUT WORK_DIR phải nằm trong training_code/ để dễ quản lý
    WORK_DIR = "augmented_dataset" 

    # Training settings
    TRAIN_RATIO = 0.8
    IMG_EXT = "jpg"
    FRAME_STEP = 1
    NUM_WORKERS = 8

    # Model settings
    MODEL_WEIGHTS = "yolov8s-worldv2.pt"  # 12.7M params
    CLASS_NAMES = ["target"]
    PARAM_LIMIT = 50_000_000  # 50M params limit

    # Masked training settings
    ENABLE_MASKING = True
    # Số lượng version data augmentation cho mỗi video/phase
    NUM_AUGMENTATIONS_PER_PHASE = 5 

    # Curriculum Learning settings
    CURRICULUM = {
        'phase1': {'epochs': (0, 15), 'mask_ratio': 0.1, 'strategy': 'random'},
        'phase2': {'epochs': (15, 35), 'mask_ratio': 0.3, 'strategy': 'span'},
        'phase3': {'epochs': (35, 50), 'mask_ratio': 0.5, 'strategy': 'keyframe'}
    }

    # BẮT BUỘC: Cố định SEED
    SEED = 42

config = Config()

def set_seed(seed=config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Các cài đặt PyTorch bắt buộc để đảm bảo tính reproduce
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# ============================================
# II. UTILITIES (Lấy từ notebook)
# ============================================

# --- 1. MASKING STRATEGIES (Chiến lược che frames) ---
class FrameMaskingStrategy:
    # (Copy nguyên các hàm staticmethod: random_mask, span_mask, keyframe_mask, block_mask)
    # Để tiết kiệm không gian, tôi sẽ chỉ để lại nội dung chính:

    @staticmethod
    def random_mask(total_frames, mask_ratio=0.3):
        if total_frames < 3: return []
        num_mask = max(1, int(total_frames * mask_ratio))
        num_mask = min(num_mask, total_frames - 2)
        if num_mask <= 0: return []
        try:
            masked_indices = random.sample(range(1, total_frames-1), num_mask)
            return sorted(masked_indices)
        except ValueError:
            return []

    @staticmethod
    def span_mask(total_frames, mask_ratio=0.3):
        if total_frames < 5: return FrameMaskingStrategy.random_mask(total_frames, mask_ratio)
        span_length = max(2, int(total_frames * 0.1))
        num_spans = max(1, int(total_frames * mask_ratio / span_length))
        masked_indices = []
        for _ in range(num_spans):
            start = random.randint(1, max(2, total_frames - span_length - 1)) if total_frames - span_length - 1 > 1 else 1
            masked_indices.extend(range(start, min(start + span_length, total_frames-1)))
        return sorted(list(set(masked_indices)))

    @staticmethod
    def keyframe_mask(frame_boxes, mask_ratio=0.3):
        if len(frame_boxes) < 3: return []
        motion_scores = []
        frame_indices = sorted(frame_boxes.keys())
        # ... logic tính toán motion_scores và chọn top frames ...
        for i in range(1, len(frame_indices) - 1):
             prev_frame = frame_indices[i-1]
             curr_frame = frame_indices[i]
             if not frame_boxes.get(prev_frame) or not frame_boxes.get(curr_frame): continue
             
             try:
                 prev_box = frame_boxes[prev_frame][0]
                 curr_box = frame_boxes[curr_frame][0]
                 
                 # Calculate center displacement
                 prev_cx = (prev_box['x1'] + prev_box['x2']) / 2
                 prev_cy = (prev_box['y1'] + prev_box['y2']) / 2
                 curr_cx = (curr_box['x1'] + curr_box['x2']) / 2
                 curr_cy = (curr_box['y1'] + curr_box['y2']) / 2
                 
                 motion = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                 motion_scores.append((curr_frame, motion))
             except (IndexError, KeyError):
                 continue

        if not motion_scores: return FrameMaskingStrategy.random_mask(len(frame_indices), mask_ratio)
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        num_mask = max(1, int(len(motion_scores) * mask_ratio))
        masked_frames = [frame for frame, _ in motion_scores[:num_mask]]
        return sorted(masked_frames)

    @staticmethod
    def block_mask(total_frames, mask_ratio=0.3):
        if total_frames < 5: return FrameMaskingStrategy.random_mask(total_frames, mask_ratio)
        block_size = max(3, int(total_frames * 0.05))
        num_blocks = total_frames // block_size
        if num_blocks < 3: return FrameMaskingStrategy.random_mask(total_frames, mask_ratio)

        num_mask_blocks = max(1, int(num_blocks * mask_ratio))
        try:
            masked_blocks = random.sample(range(1, num_blocks-1), min(num_mask_blocks, num_blocks-2))
        except ValueError: return []

        masked_indices = []
        for block_idx in masked_blocks:
            start = block_idx * block_size
            end = min(start + block_size, total_frames - 1)
            masked_indices.extend(range(start, end))
        return sorted(masked_indices)


# --- 2. TEMPORAL INTERPOLATION (Nội suy Bounding Boxes) ---
def interpolate_boxes(frame_boxes, masked_frames, method='cubic'):
    all_frame_indices = sorted(frame_boxes.keys())
    visible_frames = [f for f in all_frame_indices if f not in masked_frames]

    if len(visible_frames) < 2: return {}, all_frame_indices

    ground_truth = {}
    visible_data = [(f, frame_boxes[f][0]) for f in visible_frames
                    if frame_boxes[f] and len(frame_boxes[f]) > 0]

    if len(visible_data) < 2: return {}, visible_frames

    try:
        frames = [d[0] for d in visible_data]
        # ... logic trích xuất tọa độ và tạo hàm nội suy interp1d ...
        x1_vals = [d[1]['x1'] for d in visible_data]
        y1_vals = [d[1]['y1'] for d in visible_data]
        x2_vals = [d[1]['x2'] for d in visible_data]
        y2_vals = [d[1]['y2'] for d in visible_data]

        kind = 'linear' if len(frames) == 2 else method
        f_x1 = interp1d(frames, x1_vals, kind=kind, bounds_error=False, fill_value=(x1_vals[0], x1_vals[-1]))
        f_y1 = interp1d(frames, y1_vals, kind=kind, bounds_error=False, fill_value=(y1_vals[0], y1_vals[-1]))
        f_x2 = interp1d(frames, x2_vals, kind=kind, bounds_error=False, fill_value=(x2_vals[0], x2_vals[-1]))
        f_y2 = interp1d(frames, y2_vals, kind=kind, bounds_error=False, fill_value=(y2_vals[0], y2_vals[-1]))

        # Generate ground truth for masked frames
        for frame_idx in masked_frames:
            if frame_idx in frame_boxes:
                ground_truth[frame_idx] = [{
                    'frame': frame_idx,
                    'x1': int(np.clip(f_x1(frame_idx), 0, 10000)),
                    'y1': int(np.clip(f_y1(frame_idx), 0, 10000)),
                    'x2': int(np.clip(f_x2(frame_idx), 0, 10000)),
                    'y2': int(np.clip(f_y2(frame_idx), 0, 10000))
                }]
    except Exception as e:
        print(f"⚠️ Interpolation failed: {e}")
        return {}, visible_frames

    return ground_truth, visible_frames

# --- 3. REMOVE DUPLICATE BOXES (Loại bỏ box trùng lặp) ---
def remove_duplicate_boxes(boxes, iou_threshold=0.95):
    # (Copy nguyên logic tính IoU và loại bỏ box)
    if len(boxes) <= 1: return boxes
    
    def calculate_iou(box1, box2):
        # ... logic tính IoU ...
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min: return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    boxes_xyxy = [(bb['x1'], bb['y1'], bb['x2'], bb['y2']) for bb in boxes]
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]: continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]: continue
            iou = calculate_iou(boxes_xyxy[i], boxes_xyxy[j])
            if iou > iou_threshold:
                area_i = (boxes[i]['x2'] - boxes[i]['x1']) * (boxes[i]['y2'] - boxes[i]['y1'])
                area_j = (boxes[j]['x2'] - boxes[j]['x1']) * (boxes[j]['y2'] - boxes[j]['y1'])
                if area_i >= area_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [boxes[i] for i in range(len(boxes)) if keep[i]]

# --- 4. CURRICULUM CONTROLLER (Để giám sát) ---
class CurriculumController:
    """Điều khiển curriculum learning qua các epochs (chỉ để giám sát)"""
    def __init__(self, curriculum_config):
        self.curriculum = curriculum_config
        self.phases = sorted(curriculum_config.items(),
                            key=lambda x: x[1]['epochs'][0])

    def get_phase(self, epoch):
        for phase_name, phase_config in self.phases:
            start_epoch, end_epoch = phase_config['epochs']
            if start_epoch <= epoch < end_epoch:
                return phase_name, phase_config
        if self.phases:
            return self.phases[-1][0], self.phases[-1][1]
        return None, None

    def get_config(self, epoch):
        phase_name, phase_config = self.get_phase(epoch)
        if phase_config is None:
            return {'phase': 'unknown', 'mask_ratio': 0.0, 'strategy': 'random'}
        return {
            'phase': phase_name,
            'mask_ratio': phase_config['mask_ratio'],
            'strategy': phase_config['strategy']
        }

# ============================================
# III. DATA PREPARATION (Lấy frames và nhãn)
# ============================================

def extract_frames_with_masking(
    video_id, ann_dict, mode="train", augmentation_id=0,
    mask_strategy='random', mask_ratio=0.3
):
    """Extract frames + labels với masked frame augmentation"""

    video_dir = os.path.join(config.SAMPLES_DIR, video_id)
    video_path = os.path.join(video_dir, "drone_video.mp4")

    if not os.path.exists(video_path):
        return {'status': 'missing_video', 'video_id': video_id}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {'status': 'cannot_open', 'video_id': video_id}

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Xây dựng bbox_dict với deduplication (như trong notebook)
    bbox_dict = {}
    for interval in ann_dict.get(video_id, {}).get("annotations", []):
        for bb in interval.get("bboxes", []):
            frame_idx = bb["frame"]
            bbox_dict.setdefault(frame_idx, []).append(bb)
    
    for frame_idx in bbox_dict:
        bbox_dict[frame_idx] = remove_duplicate_boxes(bbox_dict[frame_idx], iou_threshold=0.95)


    # Áp dụng chiến lược masking
    masked_frames = []
    ground_truth = {}
    if config.ENABLE_MASKING and mask_ratio > 0 and len(bbox_dict) > 4:
        masker = FrameMaskingStrategy()
        frame_indices = sorted(bbox_dict.keys())
        num_frames_with_labels = len(frame_indices)
        
        # Chọn chiến lược mask
        if mask_strategy == 'random':
            masked_idx_list = masker.random_mask(num_frames_with_labels, mask_ratio)
        elif mask_strategy == 'span':
            masked_idx_list = masker.span_mask(num_frames_with_labels, mask_ratio)
        elif mask_strategy == 'keyframe':
            masked_frames = masker.keyframe_mask(bbox_dict, mask_ratio)
            masked_idx_list = []
        elif mask_strategy == 'block':
            masked_idx_list = masker.block_mask(num_frames_with_labels, mask_ratio)
        else:
            masked_idx_list = []
            
        # Chuyển đổi index list (0, 1, 2...) sang frame index thực
        if masked_idx_list:
            masked_frames = [frame_indices[i] for i in masked_idx_list if i < len(frame_indices)]

        # Generate ground truth for masked frames
        if masked_frames:
            ground_truth, _ = interpolate_boxes(bbox_dict, masked_frames, method='cubic')

    # Output directories
    if mode == "train":
        img_out = os.path.join(config.WORK_DIR, 'train', 'images')
        lbl_out = os.path.join(config.WORK_DIR, 'train', 'labels')
    else:
        img_out = os.path.join(config.WORK_DIR, 'val', 'images')
        lbl_out = os.path.join(config.WORK_DIR, 'val', 'labels')

    saved = 0
    masked_count = 0
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if idx % config.FRAME_STEP == 0 and idx in bbox_dict:
                img_name = f"{video_id}_aug{augmentation_id:04d}_frame_{idx:06d}.{config.IMG_EXT}"
                img_path = os.path.join(img_out, img_name)
                txt_path = os.path.join(lbl_out, img_name.replace(config.IMG_EXT, "txt"))

                # Sử dụng ground truth cho frame bị mask
                if idx in masked_frames and idx in ground_truth:
                    boxes_to_save = ground_truth[idx]
                    masked_count += 1
                elif idx in bbox_dict:
                    boxes_to_save = bbox_dict[idx]
                else:
                    boxes_to_save = []

                if boxes_to_save:
                    # Lưu ảnh (ảnh gốc)
                    if not cv2.imwrite(img_path, frame): continue 

                    lines = []
                    for bb in boxes_to_save:
                        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
                        
                        # Validate và clip box
                        if x2 <= x1 or y2 <= y1: continue
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Convert sang YOLO format (normalized cx, cy, w, h)
                        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                        bw, bh = (x2 - x1) / w, (y2 - y1) / h
                        
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1): continue
                        
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    with open(txt_path, "w") as f:
                        f.write("\n".join(lines))
                    saved += 1
            idx += 1
    finally:
        cap.release()

    return {
        'status': 'success', 'video_id': video_id, 'aug_id': augmentation_id,
        'frames_saved': saved, 'masked_frames': masked_count,
    }


# ============================================
# IV. MAIN PIPELINE
# ============================================

def main_pipeline():
    """Main training pipeline với mixed dataset và curriculum learning."""
    set_seed(config.SEED) # Set seed cố định
    
    print("🚀 ENHANCED YOLO WORLD TRAINING PIPELINE (MIXED DATA)")
    
    # 1. Load annotations
    with open(config.ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    video_ids = [a["video_id"] for a in annotations]
    ann_dict = {a["video_id"]: a for a in annotations}

    # Split train/val
    random.seed(config.SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * config.TRAIN_RATIO)
    train_videos = video_ids[:split_idx]
    val_videos = video_ids[split_idx:]

    print(f"✅ Loaded {len(video_ids)} videos. Train: {len(train_videos)}, Val: {len(val_videos)}")

    # Setup curriculum
    curriculum = CurriculumController(config.CURRICULUM)

    # 2. Tạo thư mục output và data.yaml
    os.makedirs(config.WORK_DIR, exist_ok=True)
    for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(os.path.join(config.WORK_DIR, subdir), exist_ok=True)

    # 3. Tạo data augmentation (sử dụng ThreadPoolExecutor)
    futures = []
    stats_list = []
    global_aug_id_counter = 0

    print(f"\n⚙️ Preparing data augmentations using {config.NUM_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as ex:

        # TRAIN data: Tạo nhiều version (NUM_AUGMENTATIONS_PER_PHASE) cho mỗi phase
        for phase_name, phase_config in curriculum.phases:
            mask_ratio = phase_config['mask_ratio']
            mask_strategy = phase_config['strategy']
            print(f"  -> Submitting jobs for Phase {phase_name.upper()} (Mask: {mask_ratio*100}%, Strategy: {mask_strategy})")

            for vid in train_videos:
                for _ in range(config.NUM_AUGMENTATIONS_PER_PHASE):
                    futures.append(
                        ex.submit(
                            extract_frames_with_masking,
                            vid, ann_dict, "train",
                            global_aug_id_counter,
                            mask_strategy, mask_ratio
                        )
                    )
                    global_aug_id_counter += 1

        # VAL data: Chỉ 1 version, không mask
        for vid in val_videos:
            futures.append(
                ex.submit(
                    extract_frames_with_masking,
                    vid, ann_dict, "val",
                    0, 'random', 0.0 # mask_ratio = 0.0
                )
            )

        # Thu thập kết quả
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            stats_list.append(result)
            if i % 50 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] Processed {result['video_id']}_aug{result['aug_id']:04d}. Saved {result['frames_saved']} frames.")

    # In thống kê tổng
    success_stats = [s for s in stats_list if s['status'] == 'success']
    total_frames = sum(s['frames_saved'] for s in success_stats)
    total_masked = sum(s['masked_frames'] for s in success_stats)
    print(f"\n📈 TOTAL DATASET: {total_frames} frames saved ({total_masked} masked)")

    # 4. Create data.yaml
    data_yaml = {
        "train": os.path.abspath(os.path.join(config.WORK_DIR, 'train', 'images')),
        "val": os.path.abspath(os.path.join(config.WORK_DIR, 'val', 'images')),
        "nc": 1,
        "names": config.CLASS_NAMES
    }
    data_path = os.path.join(config.WORK_DIR, "data.yaml")
    with open(data_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\n📄 data.yaml created at: {data_path}")
    print(open(data_path).read())

    # 5. Training
    print("\n🚀 STARTING TRAINING...")

    model = YOLOWorld(config.MODEL_WEIGHTS)
    model.set_classes(config.CLASS_NAMES)
    
    # Custom training callback (chỉ để giám sát curriculum phase)
    monitor = CurriculumController(config.CURRICULUM)
    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1
        phase_config = monitor.get_config(epoch)
        if epoch == 1 or (epoch) % 5 == 0 or epoch == trainer.epochs:
            print(f"\n-- 📚 Epoch {epoch}/{trainer.epochs} -- Curriculum Phase (Reference): {phase_config['phase']} --")
    
    model.add_callback("on_epoch_end", on_epoch_end)

    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=896,
        batch=32,
        lr0=5e-4,
        optimizer="AdamW",
        box=10.0,
        cls=0.5,
        dfl=1.5,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        weight_decay=0.0005,
        patience=15,
        save_period=5,
        workers=8,
        close_mosaic=10,
        project=os.path.join(config.WORK_DIR, "runs"),
        name="curriculum_MIXED_training",
        exist_ok=True,
        pretrained=True,
        verbose=True
    )

    print("\n✅ TRAINING COMPLETE!")
    best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    print(f"Best model saved to: {best_model_path}")
    
    # Chú ý: Cần copy 'best.pt' này vào thư mục /code/saved_models/ (thay tên thành models.safetensors hoặc giữ nguyên) 
    # trước khi commit docker.
    return best_model_path


if __name__ == "__main__":
    try:
        main_pipeline()
    except Exception as e:
        print(f"\n❌ An error occurred during the training pipeline: {e}")
        import traceback
        traceback.print_exc()