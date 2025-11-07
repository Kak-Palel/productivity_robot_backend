import os
from src.pose_estimator import PoseEstimator
from src.hollistic_estimator import HollisticEstimator
from src.emotion_classifier import EmotionClassifier
from ultralytics import YOLO
from juxtapose.utils.plotting import Annotator
import mediapipe as mp
import cv2 as cv
import numpy as np
import os


VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'train', 'final_classifier', 'dataset')
DETECTION_MODEL = "yolov8m.engine"

labels = os.listdir(VIDEOS_DIR)
detection_engine_path = os.path.join(os.path.dirname(__file__), '..', 'model', DETECTION_MODEL)

object_detector = YOLO("model/yolov8m.pt")
holistic = HollisticEstimator(static_image_mode=True, model_complexity=1, enable_segmentation=False, refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_annotator = Annotator()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

for label in labels:
    label_dir = os.path.join(VIDEOS_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    video_files = [f for f in os.listdir(label_dir) if f.endswith('.webm')]
    for video_file in video_files:
        video_path = os.path.join(label_dir, video_file)
        print(f'Processing {video_path} with label {label}')
        # prepare CSV for this label
        csv_path = os.path.join(VIDEOS_DIR, f"{label}.csv")
        write_header = not os.path.exists(csv_path)
        import csv

        # build header: video,frame,person_id,x1,y1,x2,y2, then pose (33*2) and face (468*2)
        pose_n = 33
        face_n = 468
        header = ['video', 'frame', 'person_id', 'x1', 'y1', 'x2', 'y2']
        for i in range(pose_n):
            header += [f'pose_{i}_x', f'pose_{i}_y']
        for i in range(face_n):
            header += [f'face_{i}_x', f'face_{i}_y']

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print('Cannot open video', video_path)
            continue

        frame_idx = 0
        SKIP_FRAMES = 5  # sample every Nth frame to limit dataset size; tune as needed

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % SKIP_FRAMES != 0:
                    frame_idx += 1
                    continue

                # run detection
                results = object_detector(frame)
                # ultralytics Results object: take first batch
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback if different API (pure numpy)
                    boxes = results[0].boxes.xyxy.numpy()
                    scores = results[0].boxes.conf.numpy()
                    class_ids = results[0].boxes.cls.numpy().astype(int)

                person_id = 0
                for box, score, class_id in zip(boxes, scores, class_ids):
                    if class_id != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    # clamp
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w - 1, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size == 0:
                        continue

                    # run MediaPipe Holistic on the crop
                    img_rgb = cv.cvtColor(person_img, cv.COLOR_BGR2RGB)
                    img_rgb.flags.writeable = False
                    res = holistic.holistic.process(img_rgb)
                    img_rgb.flags.writeable = True

                    # extract pose landmarks (33) normalized to crop
                    if res and res.pose_landmarks:
                        lm_pose = res.pose_landmarks.landmark
                        pose_kpts = np.array([[(p.x, p.y) for p in lm_pose]], dtype=np.float32)[0]  # (33,2)
                        pose_kpts = np.clip(pose_kpts, 0.0, 1.0)
                    else:
                        pose_kpts = np.zeros((pose_n, 2), dtype=np.float32)

                    # extract face landmarks (468) normalized to crop
                    if res and res.face_landmarks:
                        lm_face = res.face_landmarks.landmark
                        face_kpts = np.array([[(p.x, p.y) for p in lm_face]], dtype=np.float32)[0]  # (468,2)
                        face_kpts = np.clip(face_kpts, 0.0, 1.0)
                    else:
                        face_kpts = np.zeros((face_n, 2), dtype=np.float32)

                    # Build row: video, frame, person_id, bbox, then flattened kpts
                    row = [video_file, frame_idx, person_id, x1, y1, x2, y2]
                    # Pose and face are already normalized relative to the bbox (crop resized to 1x1),
                    # convert to float32 and flatten
                    row += [float(x) for tup in pose_kpts for x in tup]
                    row += [float(x) for tup in face_kpts for x in tup]

                    writer.writerow(row)
                    person_id += 1

                frame_idx += 1

        cap.release()
