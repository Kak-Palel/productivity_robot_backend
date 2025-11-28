"""HolisticEstimator

This module provides a small `HollisticEstimator` class (note the spelling)
that runs MediaPipe Holistic on cropped person images and returns
normalized keypoints plus a tiny placeholder emotion vector.

It is intentionally lightweight and designed to be imported from `app.py`.
The original top-level Holistic example from the repo was removed here so
the file can be imported safely; the core MediaPipe logic is encapsulated
in the class and preserves the same preprocessing/postprocessing steps
used in the example (writeable flags, RGB conversion, same drawing helpers).
"""

from pathlib import Path
import cv2 as cv
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required for hollistic_estimator.py. Install with `pip install mediapipe`") from e

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


class HollisticEstimator:
    """Run MediaPipe Holistic on person crops and return normalized keypoints.

    infer(img) -> (keypoints, scores, emotion)
      - img: BGR numpy array (cropped person)
      - keypoints: (1, 33, 2) float32, normalized x,y in [0..1] relative to the crop
      - scores: (1, 33) confidence placeholders (1.0 when landmark present)
      - emotion: (1, 4) placeholder probabilities for [productive, not_productive, tired, mad]

    This keeps the same MediaPipe preprocess steps from the example: mark
    image writeable=False before processing, convert BGR->RGB, then back.
    """

    def __init__(
        self,
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def infer(self, img: np.ndarray):
        """Run inference on a single BGR crop and return normalized keypoints.

        Normalization: MediaPipe landmarks are provided in normalized image
        coordinates (0..1) already; we keep that and return an array shaped
        (1, 33, 2). If no landmarks are detected we return zeros.

        Returns:
          kpts: (1,33,2) float32
          scores: (1,33) float32
        """
        if img is None or img.size == 0:
            return np.zeros((1, 33, 2), dtype=np.float32), np.zeros((1, 33), dtype=np.float32)

        h, w = img.shape[:2]

        # follow the example performance trick
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.holistic.process(img_rgb)

        # restore writeable (not strictly necessary for crops but consistent)
        # and convert back if caller expects to reuse the image
        img_rgb.flags.writeable = True

        # pose landmarks (33 points) in normalized coords
        if results and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            kpts = np.array([[ (p.x, p.y) for p in lm ]], dtype=np.float32)  # shape (1,33,2)
            # clamp to [0,1]
            kpts = np.clip(kpts, 0.0, 1.0)
            scores = np.ones((1, kpts.shape[1]), dtype=np.float32)
        else:
            kpts = np.zeros((1, 33, 2), dtype=np.float32)
            scores = np.zeros((1, 33), dtype=np.float32)

        return kpts, scores

    def close(self):
        try:
            self.holistic.close()
        except Exception:
            pass


if __name__ == '__main__':
    print('HollisticEstimator module. Import HollisticEstimator from this file to run per-crop inference in app.py')
    # Dummy webcam main for quick local testing. This mimics the simple
    # MediaPipe webcam loop but uses the HollisticEstimator.infer API and
    # draws normalized keypoints back onto the full-frame for visual check.
    cap = cv.VideoCapture(0)
    est = HollisticEstimator(static_image_mode=False, model_complexity=2, enable_segmentation=False, refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def draw_all_pose_landmarks(image, pose_landmarks):
        last_landmark = None
        for landmark in pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)
            # if last_landmark is not None:
            #     last_x = int(last_landmark.x * image.shape[1])
            #     last_y = int(last_landmark.y * image.shape[0])
            #     cv.line(image, (last_x, last_y), (x, y), (255, 0, 0), 2)
            # last_landmark = landmark
        
        return image


    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use MediaPipe drawing utilities directly as requested: run the
            # internal processor and draw landmarks with explicit DrawingSpec
            # objects to avoid NoneType issues in some environments.
            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = est.holistic.process(img_rgb)
            # results = est.infer(frame)
            img_rgb.flags.writeable = True

            # draw face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
                )

            # draw pose landmarks
            if results.pose_landmarks:
                print(results.pose_landmarks)
                print("len:", len(results.pose_landmarks.landmark))
                # mp_drawing.draw_landmarks(
                #     frame,
                #     results.pose_landmarks,
                #     mp_holistic.POSE_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                #     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2),
                # )

                frame = draw_all_pose_landmarks(frame, results.pose_landmarks)

            # draw left/right hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
                )

            cv.imshow('Hollistic webcam (dummy main)', cv.flip(frame, 1))
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
        est.close()
 
