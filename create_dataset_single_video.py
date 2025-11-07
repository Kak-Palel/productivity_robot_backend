import argparse
import os
import shutil
import subprocess
import tempfile
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from src.hollistic_estimator import HollisticEstimator


def open_video_with_fallback(path: str):
    os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")
    cap = cv.VideoCapture(path, cv.CAP_FFMPEG)
    tmp_transcoded = None
    if not cap.isOpened():
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tmp_path = tmp.name
            tmp.close()
            # try remux first
            cmd_remux = [ffmpeg_path, '-y', '-i', path, '-c', 'copy', tmp_path]
            try:
                subprocess.run(cmd_remux, check=True, capture_output=True, text=True, timeout=120)
                tmp_transcoded = tmp_path
                cap = cv.VideoCapture(tmp_path, cv.CAP_FFMPEG)
            except Exception:
                # remux failed; try re-encode
                cmd_reencode = [
                    ffmpeg_path, '-y', '-i', path,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-movflags', 'faststart', tmp_path
                ]
                try:
                    subprocess.run(cmd_reencode, check=True, capture_output=True, text=True, timeout=300)
                    tmp_transcoded = tmp_path
                    cap = cv.VideoCapture(tmp_path, cv.CAP_FFMPEG)
                except Exception as e:
                    print('ffmpeg fallback failed:', e)
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    return cap, tmp_transcoded


def append_video_to_csv(video_path: str, label: str, csv_path: str, skip_frames: int = 5):
    # models / estimators
    detector = YOLO('model/yolov8m.pt')
    holistic = HollisticEstimator(static_image_mode=True, model_complexity=1, enable_segmentation=False, refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    pose_n = 33
    face_n = 468

    # ensure header exists
    header_needed = not os.path.exists(csv_path)
    if header_needed:
        import csv as _csv
        with open(csv_path, 'w', newline='') as fh:
            writer = _csv.writer(fh)
            header = ['video', 'frame', 'person_id', 'x1', 'y1', 'x2', 'y2']
            for i in range(pose_n):
                header += [f'pose_{i}_x', f'pose_{i}_y']
            for i in range(face_n):
                header += [f'face_{i}_x', f'face_{i}_y']
            writer.writerow(header)

    cap, tmp_transcoded = open_video_with_fallback(video_path)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {video_path}')

    import csv
    frame_idx = 0
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # detection
            results = detector(frame)
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            except Exception:
                boxes = results[0].boxes.xyxy.numpy()
                class_ids = results[0].boxes.cls.numpy().astype(int)

            person_id = 0
            for box, class_id in zip(boxes, class_ids):
                if class_id != 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
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

                img_rgb = cv.cvtColor(person_img, cv.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                res = holistic.holistic.process(img_rgb)
                img_rgb.flags.writeable = True

                if res and res.pose_landmarks:
                    lm_pose = res.pose_landmarks.landmark
                    pose_kpts = np.array([[(p.x, p.y) for p in lm_pose]], dtype=np.float32)[0]
                    pose_kpts = np.clip(pose_kpts, 0.0, 1.0)
                else:
                    pose_kpts = np.zeros((pose_n, 2), dtype=np.float32)

                if res and res.face_landmarks:
                    lm_face = res.face_landmarks.landmark
                    face_kpts = np.array([[(p.x, p.y) for p in lm_face]], dtype=np.float32)[0]
                    face_kpts = np.clip(face_kpts, 0.0, 1.0)
                else:
                    face_kpts = np.zeros((face_n, 2), dtype=np.float32)

                row = [os.path.basename(video_path), frame_idx, person_id, x1, y1, x2, y2]
                row += [float(x) for tup in pose_kpts for x in tup]
                row += [float(x) for tup in face_kpts for x in tup]
                writer.writerow(row)
                person_id += 1

            frame_idx += 1

    cap.release()
    if tmp_transcoded:
        try:
            os.unlink(tmp_transcoded)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Append single video to dataset CSV')
    parser.add_argument('--video', '-v', required=True, help='Path to input video')
    parser.add_argument('--label', '-l', required=True, help='Label name (used only for logging)')
    parser.add_argument('--csv', '-c', required=True, help='Path to CSV to append')
    parser.add_argument('--skip-frames', '-s', type=int, default=5, help='Sample every Nth frame')
    args = parser.parse_args()

    append_video_to_csv(args.video, args.label, args.csv, skip_frames=args.skip_frames)


if __name__ == '__main__':
    main()