import kagglehub
import os
import shutil
import cv2 as cv
import mediapipe as mp
import mediapipe
import numpy as np
import csv as _csv

if not os.path.exists("dataset"):
    # Download latest version
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    os.mkdir("dataset")
    shutil.move(path, "dataset")
    os.rmdir(path=path)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

TRAIN_IMAGES_DIR = "/home/olel/Projects/productivity_robot_backend/train/emotion_classifier/dataset/train"
VALIDATION_IMAGES_DIR = "/home/olel/Projects/productivity_robot_backend/train/emotion_classifier/dataset/validation"

TRAIN_CSV_PATH = "train.csv"
VALIDATION_CSV_PATH = "validation.csv"

if __name__ == "__main__":

    TRAIN_CSV_PATH = VALIDATION_CSV_PATH
    TRAIN_IMAGES_DIR = VALIDATION_IMAGES_DIR

    if not os.path.exists(TRAIN_CSV_PATH):
        with open(TRAIN_CSV_PATH, mode="w", newline="") as train_csv_file:
            writer = _csv.writer(train_csv_file)
            header = ["label"].append(f"keypoint_{i}" for i in range(468 * 3))
            writer.writerow(header)

    with open(TRAIN_CSV_PATH, mode="a", newline="") as train_csv_file:
        writer = _csv.writer(train_csv_file)
        for label_name in os.listdir(TRAIN_IMAGES_DIR):
            label_dir = os.path.join(TRAIN_IMAGES_DIR, label_name)
            if not os.path.isdir(label_dir):
                continue

            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = cv.imread(image_path)
                if image is None:
                    continue
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ) as face_mesh:
                    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    if not results.multi_face_landmarks:
                        continue

                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     results.multi_face_landmarks[0],
                    #     mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    # )

                    face_landmarks = results.multi_face_landmarks[0]
                    keypoints = []
                    for landmark in face_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                row = [label_name] + keypoints
                
                # print(f"processed data '{label_name}' from {image_name}")

                image = cv.resize(image, (480, 480))
                keypoints_np = np.array(keypoints).reshape(478, 3)
                for point in keypoints_np:
                    x = int(point[0]*480)
                    y = int(point[1]*480)
                    cv.circle(image, (x, y), 1, (0, 0, 125 - point[2]*480), -1)
                cv.imshow(label_name, image)
                cv.waitKey(3000)
                # writer.writerow(row)