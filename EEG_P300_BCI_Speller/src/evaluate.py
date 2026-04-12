import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from preprocess import get_clean_data, run_preprocessing_fold
from features import apply_xdawn, extract_riemannian_covariances, extract_p300_features
from models import get_lda_pipeline, get_svm_pipeline, get_eegnet_pipeline, get_riemannian_pipeline
from utils import setup_environment, get_symbol_itr, get_character_prediction
import config

def run_benchmarking():
    all_summary = []

    for ds_name in config.DATASETS:
        for subj in config.TEST_SUBJECTS:
            print(f"\n--- [ {ds_name} ] Subject {subj} ---")

            try:
                epochs, X, y = get_clean_data(ds_name, subj)
            except Exception as e:
                print(f"  Skipping subject {subj}: {e}")
                continue

            skf = StratifiedGroupKFold(n_splits=5)
            groups = epochs.metadata['char_id'].values

            n_chans = X.shape[1]
            n_times = X.shape[2]

            models_list = [
                ("LDA",           get_lda_pipeline()),
                ("SVM",           get_svm_pipeline()),
                ("Xdawn+LDA",     get_lda_pipeline()),
                ("Riemannian MDM", get_riemannian_pipeline()),
                ("EEGNet",        get_eegnet_pipeline(in_chans=n_chans, input_window_samples=n_times)),
            ]

            for name, clf in models_list:
                print(f"  Training {name}...")
                metrics        = []
                subject_probs  = []
                subject_y      = []
                subject_flashes = []

                for train_idx, test_idx in skf.split(X, y, groups=groups):
                    e_tr = epochs[train_idx].copy()
                    e_te = epochs[test_idx].copy()
                    y_tr = y[train_idx]
                    y_te = y[test_idx]

                    # Per-fold AutoReject only (ICA already applied on raw)
                    e_tr, e_te = run_preprocessing_fold(e_tr, e_te)

                    # Feature extraction per model type
                    if "Xdawn" in name:
                        X_tr, X_te = apply_xdawn(e_tr, y_tr, e_te)

                    elif "Riemannian" in name:
                        X_tr = extract_riemannian_covariances(e_tr.get_data())
                        X_te = extract_riemannian_covariances(e_te.get_data())

                    elif "EEGNet" in name:
                        raw_tr = e_tr.get_data()
                        raw_te = e_te.get_data()
                        mu, sd = np.mean(raw_tr), np.std(raw_tr)
                        X_tr = ((raw_tr - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
                        X_te = ((raw_te - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
                        y_tr = y_tr.astype(np.int64)
                        y_te = y_te.astype(np.int64)

                    else:
                        # LDA / SVM: decimate to reduce dimensionality (spec: ~30 features/ch)
                        X_tr = extract_p300_features(e_tr.get_data(), decimation_factor=config.DECIMATION_FACTOR)
                        X_te = extract_p300_features(e_te.get_data(), decimation_factor=config.DECIMATION_FACTOR)

                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                    subject_probs.extend(clf.predict_proba(X_te)[:, 1])
                    subject_y.extend(y_te)
                    subject_flashes.extend(e_te.metadata['flash_id'].values)

                    metrics.append([
                        accuracy_score(y_te, y_pred),
                        recall_score(y_te, y_pred, average='binary'),
                        precision_score(y_te, y_pred, average='binary', zero_division=0),
                        f1_score(y_te, y_pred, average='binary', zero_division=0),
                    ])

                avg_m = np.mean(metrics, axis=0)
                char_acc = get_character_prediction(
                    np.array(subject_probs), np.array(subject_y), np.array(subject_flashes)
                )
                itr = get_symbol_itr(36, char_acc, dur=config.TRIAL_DURATION)

                # Confusion matrix
                y_pred_all = (np.array(subject_probs) > 0.5).astype(int)
                cm = confusion_matrix(subject_y, y_pred_all)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"CM: {name} ({ds_name} S{subj})")
                plt.xlabel("Predicted"); plt.ylabel("True")
                config.RESULTS_DIR.mkdir(exist_ok=True)
                plt.savefig(config.RESULTS_DIR / f"cm_{ds_name}_S{subj}_{name}.png")
                plt.close()

                print(f"    Acc: {avg_m[0]:.3f} | Prec: {avg_m[2]:.3f} | Rec: {avg_m[1]:.3f} | "
                      f"F1: {avg_m[3]:.3f} | Char Acc: {char_acc*100:.1f}% | ITR: {itr:.2f} bpm")

                all_summary.append([
                    ds_name, subj, name,
                    avg_m[0], avg_m[3], avg_m[2], avg_m[1],   # Acc, F1, Prec, Rec
                    char_acc, itr
                ])

    if all_summary:
        df = pd.DataFrame(
            all_summary,
            columns=['Dataset', 'Subject', 'Model', 'Acc', 'F1', 'Precision', 'Recall', 'Char_Acc', 'ITR_N36']
        )
        df.to_csv(config.RESULTS_DIR / 'all_subject_results.csv', index=False)
        print(f"\n[DONE] Benchmark complete. Results saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    setup_environment()
    run_benchmarking()
