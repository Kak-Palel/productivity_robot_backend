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
        self.X = pd.concat([locked, slacking], ignore_index=True).values.astype(np.float32)
        self.y = np.concatenate([
            np.ones(len(locked)),   # working
            np.zeros(len(slacking)) # slacking
        ]).astype(np.float32)
        
        # Normalize features (zero mean, unit variance)
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-6)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class PostureClassifier(nn.Module):
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


dataset = PostureDataset("dataset/locked_in.csv", "dataset/slacking_off.csv")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PostureClassifier(dataset.X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
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

torch.save(model.state_dict(), "posture_classifier.pth")
model.eval()
new_data = torch.tensor(dataset.X[:5]).to(device)
preds = (model(new_data) > 0.5).int()
print(preds)
