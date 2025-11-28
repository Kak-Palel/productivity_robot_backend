import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn

class PostureDataset(Dataset):
    def __init__(self, locked_path, slacking_path):
        locked = pd.read_csv(locked_path)
        slacking = pd.read_csv(slacking_path)

        # Drop the first (non-numeric) column
        locked = locked.iloc[:, 1:]
        slacking = slacking.iloc[:, 1:]

        # Combine and label
        raw = pd.concat([locked, slacking], ignore_index=True).values.astype(np.float32)
        self.y = np.concatenate([
            np.ones(len(locked)),   # working
            np.zeros(len(slacking)) # slacking
        ]).astype(np.float32)

        # Compute and store normalization params (mean/std) from raw features
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-6

        # Normalize features (zero mean, unit variance)
        self.X = (raw - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
class PostureClassifierModel(nn.Module):
    """Convolutional classifier that accepts a flat feature vector and
    reshapes it into a square image-like tensor for CNN processing.

    The model pads the input vector with zeros to the next square size
    (side x side) where side = ceil(sqrt(input_size)). This keeps the
    external API unchanged (accepts a flat feature vector) while using
    convolutional feature extractors internally.
    """
    def __init__(self, input_size):
        super().__init__()
        # compute square side and padded size
        side = int(np.ceil(np.sqrt(input_size)))
        padded = side * side
        self.input_size = input_size
        self.side = side
        self.padded = padded

        # small convolutional stack
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        # compute flattened conv output size (handle small sides)
        conv_side = max(1, side // 4)
        conv_features = 32 * conv_side * conv_side

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_features, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        if x.shape[1] < self.padded:
            pad = x.new_zeros((B, self.padded - x.shape[1]))
            x = torch.cat([x, pad], dim=1)
        x = x.view(B, 1, self.side, self.side)
        x = self.conv(x)
        x = self.head(x)
        return x
    



dataset = PostureDataset("dataset/locked_in.csv", "dataset/slacking_off.csv")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = dataset.X.shape[1]
model = PostureClassifierModel(input_size).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(4):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            preds = model(X_batch)
            predicted = (preds > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'model'), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'posture_classifier.pth')
torch.save(model.state_dict(), model_path)
# save normalization params
norm_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'posture_classifier_norm.npz')
np.savez(norm_path, mean=dataset.mean, std=dataset.std)
print(f'Saved model to {model_path} and normalization to {norm_path}')
model.eval()
new_data = torch.tensor(dataset.X[:5]).to(device)
preds = (model(new_data) > 0.5).int()
print(preds)
