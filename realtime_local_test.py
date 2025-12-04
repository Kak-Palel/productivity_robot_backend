# from src.object_detector import ObjectDetector, class_names, annotate_image
from src.pose_estimator import PoseEstimator
from src.hollistic_estimator import HollisticEstimator
from src.emotion_classifier import EmotionClassifier
from src.posture_classifier import PostureClassifierWrapper
from ultralytics import YOLO
from juxtapose.utils.plotting import Annotator
import mediapipe as mp
import cv2 as cv
import numpy as np
import os
import time

CAMERA_PATH = "/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0"
DETECTION_MODEL = "yolov8m.engine"

if __name__ == '__main__':
    detection_engine_path = os.path.join(os.path.dirname(__file__), 'model', DETECTION_MODEL)

    # object_detector = ObjectDetector(detection_engine_path)
    object_detector = YOLO("model/yolov8m.pt")
    # Instantiate the MediaPipe-based holistic estimator and the emotion classifier
    # (we keep the old PoseEstimator import in case you want to switch back).
    holistic = HollisticEstimator(static_image_mode=True, model_complexity=1, enable_segmentation=False, refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    emotion_clf = EmotionClassifier()
    # load posture classifier (optional)
    try:
        posture_clf = PostureClassifierWrapper(model_path=os.path.join(os.path.dirname(__file__), 'model', 'posture_classifier_v7.pth'))
    except Exception as e:
        print('Posture classifier not available:', e)
        posture_clf = None
    pose_annotator = Annotator()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv.VideoCapture(CAMERA_PATH, cv.CAP_V4L2)
    # cap = cv.VideoCapture(0)
    # os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")
    # cap = cv.VideoCapture("/home/olel/Projects/productivity_robot_backend/test.mp4")
    # cap = cv.VideoCapture("/home/olel/Projects/productivity_robot_backend/train/final_classifier/dataset/locked_in/2025-11-06-175545.webm")
    # cap = cv.VideoCapture("/home/olel/Projects/productivity_robot_backend/train/final_classifier/dataset/slacking_off/2025-11-06-175907.webm")
    if not cap.isOpened():
        print('Cannot open camera (index 0)')
        raise SystemExit(1)
    

    try:
        while True:
            now = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = object_detector(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
            scores = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs

            for box, score, class_id in zip(boxes, scores, class_ids):
                if class_id != 0:  # person class
                    continue
                x1, y1, x2, y2 = map(int, box)
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                # Run MediaPipe Holistic on the person crop and draw landmarks
                img_rgb = cv.cvtColor(person_img, cv.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = holistic.holistic.process(img_rgb)
                img_rgb.flags.writeable = True

                # Draw landmarks directly onto the person crop using MediaPipe draw helpers
                annotated_crop = person_img.copy()
                # if results.face_landmarks:
                #     mp_drawing.draw_landmarks(
                #         annotated_crop,
                #         results.face_landmarks,
                #         mp_holistic.FACEMESH_TESSELATION,
                #         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                #         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
                #     )
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_crop,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2),
                    )
                # if results.left_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         annotated_crop,
                #         results.left_hand_landmarks,
                #         mp_holistic.HAND_CONNECTIONS,
                #         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                #         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
                #     )
                # if results.right_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         annotated_crop,
                #         results.right_hand_landmarks,
                #         mp_holistic.HAND_CONNECTIONS,
                #         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                #         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
                #     )

                # place annotated crop back into frame
                frame[y1:y2, x1:x2] = annotated_crop

                # Emotion classification: try to extract a tight face crop and call classifier
                face_emotion = None
                try:
                    if results.face_landmarks:
                        # compute bounding box from face landmarks (normalized coords)
                        lm = results.face_landmarks.landmark
                        xs = [p.x for p in lm]
                        ys = [p.y for p in lm]
                        fx1 = int(max(0, min(xs) * (x2 - x1) - 0.05 * (x2 - x1)))
                        fy1 = int(max(0, min(ys) * (y2 - y1) - 0.05 * (y2 - y1)))
                        fx2 = int(min((x2 - x1), max(xs) * (x2 - x1) + 0.05 * (x2 - x1)))
                        fy2 = int(min((y2 - y1), max(ys) * (y2 - y1) + 0.05 * (y2 - y1)))
                        # convert to absolute coords on full frame
                        abs_fx1 = x1 + fx1
                        abs_fy1 = y1 + fy1
                        abs_fx2 = x1 + fx2
                        abs_fy2 = y1 + fy2
                        face_crop = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
                        if face_crop.size != 0:
                            # write to a temp file and call the classifier (it expects a path)
                            import tempfile
                            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            tmp_path = tmp.name
                            tmp.close()
                            cv.imwrite(tmp_path, face_crop)
                            pred_emotion, pred_conf, pred_class = emotion_clf.classify(tmp_path)
                            face_emotion = (pred_emotion, pred_conf, pred_class)
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                except Exception as e:
                    # don't let emotion classifier errors break the loop
                    print('Emotion classification failed:', e)
                if face_emotion is not None:
                    em_text = f"{face_emotion[0]}:{face_emotion[1]:.2f}"
                    cv.putText(frame, em_text, (x1, max(0, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Run posture classifier if available. Build pose and face arrays
                try:
                    # pose landmarks (33)
                    if results.pose_landmarks:
                        lm_pose = results.pose_landmarks.landmark
                        pose_kpts = np.array([[ (p.x, p.y) for p in lm_pose ]], dtype=np.float32)[0]
                        pose_kpts = np.clip(pose_kpts, 0.0, 1.0)
                    else:
                        pose_kpts = np.zeros((33, 2), dtype=np.float32)

                    # face landmarks (468)
                    if results.face_landmarks:
                        lm_face = results.face_landmarks.landmark
                        face_kpts = np.array([[ (p.x, p.y) for p in lm_face ]], dtype=np.float32)[0]
                        face_kpts = np.clip(face_kpts, 0.0, 1.0)
                    else:
                        face_kpts = np.zeros((468, 2), dtype=np.float32)

                    if posture_clf is not None:
                        prob = posture_clf.infer_from_landmarks(pose_kpts, face_kpts, bbox=(x1, y1, x2, y2), frame_idx=0, person_id=0)
                        label = 'working' if prob > 0.5 else 'slacking'
                        txt = f'{label}:{prob:.2f}'
                        cv.rectangle(frame, (x1, min(y2 + 20, frame.shape[0]-5)), (x1 + cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0], min(y2 + 20, frame.shape[0]-5) - 15), (0, 255, 0) if label == 'working' else (0, 0, 255), -1)
                        cv.putText(frame, txt, (x1, min(y2 + 20, frame.shape[0]-5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                except Exception as e:
                    print('Posture classification failed:', e)

            cv.imshow('Pose estimation', frame)
            print(f'Pose estimation time: {(time.time() - now):.3f} s')

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            print(f'Frame total time: {(time.time() - now):.3f} s')
    finally:
        cap.release()
        cv.destroyAllWindows()