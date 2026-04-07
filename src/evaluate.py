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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Local Imports (Aligned with submission structure)
from preprocess import get_clean_data
from models import EEGNet

# Global Config
warnings.filterwarnings('ignore')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs('results', exist_ok=True)

# Hardware Audit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Hardware Audit: Running on {str(device).upper()} ---")
if device.type == 'cuda':
    print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")
else:
    print("--- WARNING: Running on CPU. Training will be slow. ---")

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
            subjects = [1, 2, 3] # Adjusted for verification
        else:
            subjects = [1, 2] # EPFLP300
            
        for subj in subjects:
            print(f"  - Evaluate Subject {subj}...")
            try:
                epochs, X, y = get_clean_data(dataset_name=ds_name, subj=subj)
            except Exception as e:
                print(f"    ! Error loading {ds_name} sub {subj}: {e}")
                continue

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            
            models_list = [
                ("LDA", Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])),
                ("SVM", Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True))])),
                ("EEGNet", "DL")
            ]

            for name, clf in models_list:
                fold_acc = []
                for train_idx, test_idx in skf.split(X, y):
                    if name == "EEGNet":
                        mu, sd = np.mean(X[train_idx]), np.std(X[train_idx])
                        X_tr = torch.Tensor((X[train_idx] - mu)/sd)[:, None, :, :]
                        X_te = torch.Tensor((X[test_idx] - mu)/sd)[:, None, :, :]
                        
                        loader = DataLoader(TensorDataset(X_tr, torch.LongTensor(y[train_idx])), batch_size=32, shuffle=True)
                        net = EEGNet(n_chan=X.shape[1], n_time=X_tr.shape[-1]).to(device)
                        opt = optim.Adam(net.parameters(), lr=0.001)
                        
                        # Adaptive Class Weighting
                        n_pos = np.sum(y[train_idx])
                        weight = len(y[train_idx]) / (2 * n_pos) if n_pos > 0 else 5.0
                        crit = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, weight]).to(device))

                        for _ in range(30): # Regularized training
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
                # SCIENTIFIC FIX: ITR N=2 for binary flash detection
                itr = get_itr(2, avg_m[0])
                all_summary.append([ds_name, subj, name, avg_m[0], avg_m[1], avg_m[2], avg_m[3], itr])
                print(f"    {name} -> F1: {avg_m[3]:.3f} | Acc: {avg_m[0]:.3f} | ITR: {itr:.1f}")

    # --- REPORTING ---
    df = pd.DataFrame(all_summary, columns=['Dataset', 'Subject', 'Model', 'Acc', 'Recall', 'Prec', 'F1', 'ITR'])
    df.to_csv('results/all_subject_results.csv', index=False)
    
    final_report = df.groupby(['Dataset', 'Model'])[['Acc', 'F1', 'ITR']].mean().round(3)
    print("\n" + "="*50)
    print("--- FINAL SUBMISSION BENCHMARK ---")
    print("="*50)
    print(final_report)
    print("="*50)
    print("\nFull breakdown saved to results/all_subject_results.csv")
