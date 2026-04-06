import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Local Import
from data_loader import get_clean_data

# Global Config
warnings.filterwarnings('ignore')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

class EEGNet(nn.Module):
    """Compact CNN for EEG classification (Lawhern et al., 2018)."""
    def __init__(self, n_chan=16, n_time=32):
        super(EEGNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_chan, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), groups=16, padding='same', bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.fc = nn.LazyLinear(2)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x)).view(x.size(0), -1))

def get_itr(n, acc, dur=2.0):
    if acc >= 0.99: return np.log2(n) * 60 / dur
    if acc <= 1/n: return 0
    return (np.log2(n) + acc*np.log2(acc) + (1-acc)*np.log2((1-acc)/(n-1))) * 60 / dur

if __name__ == "__main__":
    datasets = ['BNCI2014_009', 'EPFLP300']
    all_summary = []

    print("--- Starting Multi-Dataset Comparative Benchmark ---")
    
    for ds_name in datasets:
        print(f"\n>>> AUDITING DATASET: {ds_name}...")
        
        # BNCI2014_009 has 10 subjects, EPFLP300 has subjects [1,2,3,4,6,7,8,9]
        if ds_name == 'BNCI2014_009':
            subjects = [1, 2, 3] # Reduced for verification, normally range(1, 11)
        else:
            subjects = [1, 2] # EPFLP300
            
        for subj in subjects:
            print(f"  - Loading Subject {subj}...")
            try:
                epochs, X, y = get_clean_data(dataset_name=ds_name, subj=subj)
            except Exception as e:
                print(f"    ! Error loading {ds_name} sub {subj}: {e}")
                continue

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) # 3-fold for speed
            
            models = [
                ("LDA", LinearDiscriminantAnalysis()),
                ("SVM", SVC(kernel='rbf', class_weight='balanced', probability=True)),
                ("EEGNet", "DL")
            ]

            for name, clf in models:
                fold_acc = []
                for train_idx, test_idx in skf.split(X, y):
                    if name == "EEGNet":
                        # Standardize per subject
                        mu, sd = np.mean(X[train_idx]), np.std(X[train_idx])
                        X_tr = torch.Tensor((X[train_idx] - mu)/sd)[:, None, :, :]
                        X_te = torch.Tensor((X[test_idx] - mu)/sd)[:, None, :, :]
                        
                        # GPU Support
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        loader = DataLoader(TensorDataset(X_tr, torch.LongTensor(y[train_idx])), batch_size=32, shuffle=True)
                        net = EEGNet(n_chan=X.shape[1], n_time=X_tr.shape[-1]).to(device)
                        opt = optim.Adam(net.parameters(), lr=0.001)
                        # Balanced loss on device
                        crit = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 5.0]).to(device))

                        for _ in range(30):
                            net.train()
                            for b_x, b_y in loader:
                                b_x, b_y = b_x.to(device), b_y.to(device)
                                opt.zero_grad(); crit(net(b_x), b_y).backward(); opt.step()
                        
                        net.eval()
                        with torch.no_grad():
                            p = torch.argmax(net(X_te.to(device)), dim=1).cpu().numpy()
                    else:
                        X_tr = X[train_idx].reshape(len(train_idx), -1)
                        X_te = X[test_idx].reshape(len(test_idx), -1)
                        clf.fit(X_tr, y[train_idx])
                        p = clf.predict(X_te)
                    
                    fold_acc.append([
                        accuracy_score(y[test_idx], p),
                        recall_score(y[test_idx], p),
                        precision_score(y[test_idx], p),
                        f1_score(y[test_idx], p)
                    ])
                
                avg_m = np.mean(fold_acc, axis=0)
                all_summary.append([ds_name, subj, name, avg_m[0], avg_m[1], avg_m[2], avg_m[3], get_itr(36, avg_m[0])])

    # --- REPORTING ---
    df = pd.DataFrame(all_summary, columns=['Dataset', 'Subject', 'Model', 'Acc', 'Recall', 'Prec', 'F1', 'ITR'])
    df.to_csv('results/multi_dataset_results.csv', index=False)
    
    final_report = df.groupby(['Dataset', 'Model'])[['Acc', 'F1', 'ITR']].mean().round(3)
    print("\n" + "="*50)
    print("--- FINAL COMPARATIVE BENCHMARK ---")
    print("="*50)
    print(final_report)
    print("="*50)
    print("\nFull breakdown saved to results/multi_dataset_results.csv")
