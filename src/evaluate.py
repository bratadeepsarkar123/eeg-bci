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
from preprocess import get_clean_data, apply_spatial_ica
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

            skf = StratifiedKFold(n_splits=3, shuffle=False)
            
            models_list = [
                ("LDA", Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])),
                ("SVM", Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True))])),
                ("EEGNet", "DL")
            ]

            # To store fold metrics per model
            fold_metrics = {name: [] for name, _ in models_list}

            # Bug #2 & Bug #4 Fix: Split without shuffling, apply ICA fold by fold
            for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
                # ICA Spatial Audit inside CV Loop
                epochs_train_cv = epochs[train_idx].copy()
                epochs_test_cv = epochs[test_idx].copy()
                epochs_train_cv, epochs_test_cv = apply_spatial_ica(epochs_train_cv, epochs_test_cv)
                
                # Transformed features
                X_tr_ic = epochs_train_cv.get_data()
                X_te_ic = epochs_test_cv.get_data()
                
                for name, clf in models_list:
                    if name == "EEGNet":
                        mu, sd = np.mean(X_tr_ic), np.std(X_tr_ic)
                        X_tr = torch.Tensor((X_tr_ic - mu)/sd)[:, None, :, :]
                        X_te = torch.Tensor((X_te_ic - mu)/sd)[:, None, :, :]
                        
                        loader = DataLoader(TensorDataset(X_tr, torch.LongTensor(y[train_idx])), batch_size=32, shuffle=True)
                        net = EEGNet(n_chan=X_tr_ic.shape[1], n_time=X_tr.shape[-1]).to(device)
                        opt = optim.Adam(net.parameters(), lr=0.001)
                        
                        # Adaptive Class Weighting
                        n_pos = np.sum(y[train_idx])
                        weight = len(y[train_idx]) / (2 * n_pos) if n_pos > 0 else 5.0
                        crit = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, weight]).to(device))

                        for _ in range(30):
                            net.train()
                            for b_x, b_y in loader:
                                b_x, b_y = b_x.to(device), b_y.to(device)
                                opt.zero_grad(); crit(net(b_x), b_y).backward(); opt.step()
                        
                        net.eval()
                        with torch.no_grad():
                            p = torch.argmax(net(X_te.to(device)), dim=1).cpu().numpy()
                    else:
                        X_tr = X_tr_ic.reshape(len(train_idx), -1)
                        X_te = X_te_ic.reshape(len(test_idx), -1)
                        clf.fit(X_tr, y[train_idx])
                        p = clf.predict(X_te)
                    
                    fold_metrics[name].append([
                        accuracy_score(y[test_idx], p),
                        recall_score(y[test_idx], p),
                        precision_score(y[test_idx], p),
                        f1_score(y[test_idx], p)
                    ])

            # Bug #9 Fix: Remove invalid single-trial ITR calculation
            for name, _ in models_list:
                avg_m = np.mean(fold_metrics[name], axis=0)
                all_summary.append([ds_name, subj, name, avg_m[0], avg_m[1], avg_m[2], avg_m[3]])
                print(f"    {name} -> F1: {avg_m[3]:.3f} | Acc: {avg_m[0]:.3f}")

    # --- REPORTING ---
    df = pd.DataFrame(all_summary, columns=['Dataset', 'Subject', 'Model', 'Acc', 'Recall', 'Prec', 'F1'])
    df.to_csv('results/all_subject_results.csv', index=False)
    
    final_report = df.groupby(['Dataset', 'Model'])[['Acc', 'F1']].mean().round(3)
    print("\n" + "="*50)
    print("--- FINAL SUBMISSION BENCHMARK ---")
    print("="*50)
    print(final_report)
    print("="*50)
    print("\nFull breakdown saved to results/all_subject_results.csv")
