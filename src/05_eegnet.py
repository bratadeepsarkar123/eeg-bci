import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
from moabb.datasets import BNCI2014_009
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

class EEGNet(nn.Module):
    def __init__(self, n_channels=16, n_times=32, n_classes=2):
        super(EEGNet, self).__init__()
        
        # Temporal & Spatial Filtering
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        # Separable Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), groups=16, padding='same', bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Data Acquisition
ds = BNCI2014_009()
data = ds.get_data(subjects=[1])
raw = data[1]['0']['0']
raw.filter(0.1, 30.0, verbose=False)

# Preprocessing
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
epochs.decimate(8)

# Input Preparation
X = epochs.get_data()
X = (X - np.mean(X)) / np.std(X) # Standard Scaling
X = X[:, np.newaxis, :, :]
y = epochs.events[:, -1] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_set = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# Training Setup
model = EEGNet(n_channels=16, n_times=X.shape[-1])
weights = torch.Tensor([1.0, 10.0]) 
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(100):
    model.train()
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(b_x), b_y)
        loss.backward()
        optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(torch.Tensor(X_test)), dim=1).numpy()

print("--- EEGNet Performance ---")
print(classification_report(y_test, y_pred))
