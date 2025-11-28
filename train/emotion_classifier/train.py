import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv as _csv

TRAIN_CSV_PATH = "/home/olel/Projects/productivity_robot_backend/train/emotion_classifier/train.csv"
VALID_CSV_PATH = "/home/olel/Projects/productivity_robot_backend/train/emotion_classifier/validation.csv"
BATCH_SIZE = 32
EPOCHS = 80

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        print(f"[INFO] Loading dataset from {data_path}")
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # self.data_file = open(data_path, 'r')
        # self.reader = _csv.reader(self.data_file, delimiter=',', quotechar='"')
        # next(self.reader)
        # self.label = np.array([self.get_label_index(row[0]) for row in self.reader])
        
        self.data_file = open(data_path, 'r')
        self.reader = _csv.reader(self.data_file, delimiter=',', quotechar='"')
        next(self.reader)

        self.data = np.array([])
        self.label = np.array([])

        for row in self.reader:
            index = int(len(self.data)*np.random.rand())
            self.data = np.insert(self.data, index, np.array([float(x) for x in row[1:]]), axis=0) if len(self.data) > 0 else np.array([np.array([float(x) for x in row[1:]])])
            self.label = np.insert(self.label, index, self.get_label_index(row[0]))
            print(f"[DEBUG] self.label shape: {self.label.shape}, self.data shape: {self.data.shape}, total samples: {len(self.reader)}", end='\r')
        # self.data = np.array([row for row in self.reader])
        # print(f"[INFO] Shuffling dataset")
        # np.random.shuffle(self.data)
        print(self.label[:10])

        self.label = np.array([self.get_label_index(row[0]) for row in self.data])
        self.data = np.array([list(map(float, row[1:])) for row in self.data], dtype=np.float32)

        print(f"[INFO] Loaded {len(self.data)} samples from {data_path}")
        print(f"[INFO] len of labels: {len(self.label)}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def get_label_index(self, label_name):
        return self.classes.index(label_name)
    
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Unflatten(1, (1, 478, 3)),
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(45888, 32768),
            nn.ReLU(),
            nn.Linear(32768, 16384),
            nn.ReLU(),
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
    
    def forward(self, x):
        return self.layers(x)

def fit(model, epochs, train_loader, valid_loader, loss_function, optimizer, device):
    best_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in train_loader:
            if device.type == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                if device.type == 'cuda':
                    inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"[INFO] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
        torch.save(model.state_dict(), f"emotion_classifier_epoch_{epoch}_{epoch_loss}.pth")
        print(f"[INFO] Saved model checkpoint for epoch {epoch}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            
    print(f"[INFO] Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
    # torch.save(model.state_dict(), f"emotion_classifier_best_epoch_{best_epoch}_{best_loss}.pth")
            

if __name__ == "__main__":
    train_dataset = EmotionDataset(data_path=TRAIN_CSV_PATH)
    valid_dataset = EmotionDataset(data_path=VALID_CSV_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EmotionClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    fit(model, EPOCHS, train_loader=train_loader, valid_loader=valid_loader, loss_function=loss_function, optimizer=optimizer, device=device)