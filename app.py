from flask import Flask, request, jsonify
import requests
from pymongo import MongoClient
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
import datetime
import time
import threading

CAMERA_PATH = "/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0"
DETECTION_MODEL = "yolov8m.engine"
MONGODB_URI = "mongodb://localhost:27017/"
ESP32_BASE_URL = "http://10.126.127.177"

app = Flask(__name__)
mongo_client = MongoClient(MONGODB_URI)
database = mongo_client['productivity_robot']


def infer(image, object_detector, holistic, emotion_clf, posture_clf):
    """Run inference on a single image and return annotated image and results."""
    results = object_detector(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs

    inference_results = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id != 0:  # person class
            continue
        x1, y1, x2, y2 = map(int, box)
        person_img = image[y1:y2, x1:x2]
        if person_img.size == 0:
            continue

        image_rgb = cv.cvtColor(person_img, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.holistic.process(image_rgb)
        image_rgb.flags.writeable = True

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
                face_crop = image[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
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
        
        posture_prob = None
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
                posture_prob = posture_clf.infer_from_landmarks(pose_kpts, face_kpts, bbox=(x1, y1, x2, y2), frame_idx=0, person_id=0)
        except Exception as e:
            print('Posture classification failed:', e)
        
        inference_results.append(dict({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'score': float(score),
            'emotion': {
                'label': face_emotion[0] if face_emotion else None,
                'confidence': float(face_emotion[1]) if face_emotion else None,
                'class': int(face_emotion[2]) if face_emotion else None,
            } if face_emotion else None,
            'is_working': {
                'probability': float(posture_prob),
                'label': 'work' if posture_prob and posture_prob > 0.5 else 'not_work'
            } if posture_prob is not None else None
        }))
        
    return inference_results

def save_to_db(results):
    """Save inference results to MongoDB."""
    emotion_collection = database['emotion']
    productivity_collection = database['productivity']
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    if results.get('emotion') is not None:
        emotion_data = results.get('emotion')
        emotion_data['timestamp'] = timestamp.isoformat()
        emotion_collection.insert_one(emotion_data)

    if results.get('is_working') is not None:
        productivity_data = results.get('is_working')
        productivity_data['timestamp'] = timestamp.isoformat()
        productivity_collection.insert_one(productivity_data)

def inference_cron(object_detector, holistic, emotion_clf, posture_clf):
    while True:
        url = ESP32_BASE_URL + "/capture"
        print("Fetching image from ESP32 camera...")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch image from ESP32 camera. Status code: {response.status_code}")
            return
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv.imdecode(image_data, cv.IMREAD_COLOR)
        cv.imshow("Captured Image", image)
        cv.waitKey(1)
        results = infer(image, object_detector, holistic, emotion_clf, posture_clf)
        print("Inference results:", results)
        if results is not None and len(results) > 0:
            save_to_db(results[0])

        time.sleep(5)

@app.route('/api/emotions', methods=['GET'])
def get_emotion_metrics():
    limit = int(request.args.get('limit', 0))
    emotion_collection = database['emotion']
    emotions = list(emotion_collection.find().sort('timestamp', -1).limit(limit))
    for em in emotions:
        em['_id'] = str(em['_id'])
    return jsonify(emotions)

@app.route('/api/productivity', methods=['GET'])
def get_productivity_metrics():
    limit = int(request.args.get('limit', 0))
    productivity_collection = database['productivity']
    productivity = list(productivity_collection.find().sort('timestamp', -1).limit(limit))
    for p in productivity:
        p['_id'] = str(p['_id'])
    return jsonify({"entries" : productivity})

if __name__ == '__main__':
    if "emotion" not in database.list_collections():
        database.create_collection("emotion")
    if "productivity" not in database.list_collections():
        database.create_collection("productivity")

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

    schedule.every(5).seconds.do(inference_cron, object_detector, holistic, emotion_clf, posture_clf)
    thread1 = threading.Thread(target=inference_cron, args=(object_detector, holistic, emotion_clf, posture_clf))
    thread1.start()

    app.run(host='0.0.0.0', port=5000, debug=True)