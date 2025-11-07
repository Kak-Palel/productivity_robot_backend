import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class PostureClassifierModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class PostureClassifierWrapper:
    """Wrapper to load the trained posture classifier and run inference.

    It replicates the training data preprocessing: reads the train CSVs to
    compute mean/std used for feature normalization (zero-mean, unit-variance).

    Usage:
      clf = PostureClassifierWrapper(model_path='model/posture_classifier.pth')
      prob = clf.infer_from_features(feature_vector)  # returns probability in [0,1]

    The feature vector must follow the same column ordering used by
    `train/final_classifier/create_dataset.py` and `train/final_classifier/train.py`.
    Specifically: [frame, person_id, x1, y1, x2, y2, pose_0_x, pose_0_y, ..., face_467_y]
    """

    def __init__(self, model_path: str = 'model/posture_classifier.pth',
                 train_csvs=None, device=None):
        self.model_path = Path(model_path)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # default train CSVs (if available)
        if train_csvs is None:
            base = Path(__file__).resolve().parents[1] / 'train' / 'final_classifier' / 'dataset'
            locked = base / 'locked_in.csv'
            slacking = base / 'slacking_off.csv'
            train_csvs = [str(locked), str(slacking)]

        # load and compute mean/std used in training
        self._compute_normalization(train_csvs)

        # build model with inferred input size
        input_size = int(self.mean_.shape[0])
        self.model = PostureClassifierModel(input_size).to(self.device)

        if not self.model_path.exists():
            raise FileNotFoundError(f'Posture classifier not found at {self.model_path}')

        state = torch.load(str(self.model_path), map_location=self.device)
        # allow either state_dict or full model
        if isinstance(state, dict) and not any(k.startswith('model') for k in state.keys()):
            # assume it's a state_dict
            self.model.load_state_dict(state)
        else:
            # maybe saved as model.state_dict() or whole model
            try:
                self.model.load_state_dict(state)
            except Exception:
                # last resort: if saved whole model object
                loaded = state
                if hasattr(loaded, 'state_dict'):
                    self.model.load_state_dict(loaded.state_dict())

        self.model.eval()

    def _compute_normalization(self, csv_paths):
        import pandas as pd
        parts = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            parts.append(df)

        if not parts:
            raise FileNotFoundError('No training CSVs found to compute normalization')

        data = pd.concat(parts, ignore_index=True)
        # drop first (non-numeric) column as in training script
        data = data.iloc[:, 1:]
        X = data.values.astype(np.float32)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6

    def _normalize(self, x: np.ndarray):
        x = x.astype(np.float32)
        return (x - self.mean_) / self.std_

    def infer_from_features(self, features: np.ndarray):
        """Accepts a 1D numpy array of features (same order as training) and returns
        probability of the positive class (working / locked_in)."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.mean_.shape[0]:
            raise ValueError(f'Feature vector length {features.shape[1]} does not match model input {self.mean_.shape[0]}')

        xf = self._normalize(features)
        xt = torch.from_numpy(xf).to(self.device)
        with torch.no_grad():
            pred = self.model(xt).squeeze(1).cpu().numpy()
        return float(pred[0])

    def infer_from_landmarks(self, pose_kpts: np.ndarray, face_kpts: np.ndarray, bbox=None, frame_idx=0, person_id=0):
        """Builds the feature vector from landmarks and bbox then runs inference.

        pose_kpts: (33,2) normalized to bbox (values in [0,1])
        face_kpts: (468,2) normalized to bbox
        bbox: (x1,y1,x2,y2) absolute coords — included in the feature vector
        """
        if bbox is None:
            x1 = y1 = x2 = y2 = 0
        else:
            x1, y1, x2, y2 = bbox

        pose_flat = np.asarray(pose_kpts, dtype=np.float32).reshape(-1)
        face_flat = np.asarray(face_kpts, dtype=np.float32).reshape(-1)
        row = np.concatenate((np.array([frame_idx, person_id, x1, y1, x2, y2], dtype=np.float32), pose_flat, face_flat))
        return self.infer_from_features(row)
