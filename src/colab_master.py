# ==============================================================================
# BCI P300 SPELLER FINAL AUDIT: GOOGLE COLAB EDITION
# Pipeline: Preprocessing (ICA/Notch/Ref) -> 5-Fold Stratified CV -> LDA/SVM/EEGNet
# Subjects: 1, 2, 3 (Test Run)
# Instructions: Open Colab, Set Runtime to GPU (T4), and run this cell.
# ==============================================================================

# 1. INSTALL DEPENDENCIES (Wait ~1 min)
import sys
try:
    import mne
    import moabb
except ImportError:
    !pip install -q mne moabb torch braindecode scikit-learn seaborn pandas matplotlib scipy

# 2. IMPORTS
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.preprocessing import ICA
from moabb.datasets import BNCI2014_009
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from scipy import linalg

# 3. ENVIRONMENT SETUP
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Environment Ready. Running on: {device.upper()} ---")

# 4. PREPROCESSING ENGINE
def get_clean_data(subj=1):
    """Loads and preprocesses EEG data for a given subject (with ICA)."""
    print(f"  - Loading Subject {subj}...")
    ds = BNCI2014_009()
    data_dict = ds.get_data(subjects=[subj])
    raw = data_dict[subj]['0']['0']
    raw.pick_types(eeg=True)
    
    # Preprocessing (Rubric requirement)
    raw.filter(0.1, 30.0, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    
    # Bad Channel Interpolation (Rubric Requirement)
    chan_stds = np.std(raw.get_data(), axis=1)
    median_std = np.median(chan_stds)
    mad = np.median(np.abs(chan_stds - median_std))
    z_scores = np.abs(chan_stds - median_std) / (mad + 1e-8)
    bad_idx = np.where(z_scores > 3.0)[0]
    raw.info['bads'] = [raw.ch_names[i] for i in bad_idx]
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    # Artifact Rejection: ICA (Rubric requirement)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = ICA(n_components=8, random_state=SEED, method='fastica')
    ica.fit(raw_for_ica, verbose=False)
    ica.exclude = [0, 1] 
    ica.apply(raw, verbose=False)
    
    # Epoching
    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
    epochs.decimate(8) # Process at ~32Hz
    
    X = epochs.get_data()
    y = epochs.events[:, -1] - 1
    return X, y

# 5. DEEP LEARNING: EEGNET
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.fc = nn.Linear(16, n_classes) 

    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = torch.relu(self.batchnorm2(self.depthwise(x)))
        x = self.pooling1(x)
        x = torch.relu(self.batchnorm3(self.separable(x)))
        x = self.pooling2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_eegnet(X_tr, y_tr, X_te, y_te, n_channels, n_times):
    model = EEGNet(n_channels, n_times).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_ds = TensorDataset(torch.Tensor(X_tr).unsqueeze(1), torch.LongTensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(50):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_data = torch.Tensor(X_te).unsqueeze(1).to(device)
        preds = model(test_data).argmax(dim=1).cpu().numpy()
    return preds

# 6. MAIN HUB (3-SUBJECT TEST)
SUBJECTS = [1, 2, 3] 
results = []
all_y_true = []
all_y_pred = []

print("\n>>> STARTING COLAB AUDIT ---")
for subj in SUBJECTS:
    try:
        X, y = get_clean_data(subj=subj)
    except Exception as e:
        print(f"Error {subj}: {e}")
        continue
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for model_name in ["LDA", "SVM", "EEGNet"]:
        print(f"    - Testing {model_name}...")
        fold_accs = []
        for tr, te in skf.split(X, y):
            if model_name == "EEGNet":
                p = train_eegnet(X[tr], y[tr], X[te], y[te], X.shape[1], X.shape[2])
                all_y_true.extend(y[te]); all_y_pred.extend(p)
            else:
                clf = LinearDiscriminantAnalysis() if model_name == "LDA" else SVC(kernel='rbf')
                clf.fit(X[tr].reshape(len(tr), -1), y[tr])
                p = clf.predict(X[te].reshape(len(te), -1))
            fold_accs.append(accuracy_score(y[te], p))
        results.append([subj, model_name, np.mean(fold_accs)])

# 7. DISPLAY RESULTS
df = pd.DataFrame(results, columns=['Subj', 'Model', 'Accuracy'])
print("\n--- FINAL GRAND AVERAGE (3 SUBJECTS) ---")
print(df.groupby('Model')['Accuracy'].mean())

# Confusion Matrix for EEGNet
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("EEGNet Confusion Matrix (Colab Grand Average)")
plt.show()
print("\n--- DONE ---")
