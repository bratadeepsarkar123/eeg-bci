import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import get_clean_data, run_preprocessing_fold
from features import extract_p300_features
from utils import get_character_prediction, get_symbol_itr, setup_environment

# 12 flashes x 0.175s SOA
TRIAL_DURATION = 2.1
DECIMATION_FACTOR = 3


def run_ensemble_benchmark(datasets=None, subjects=None):
    """
    Ensemble Benchmark using SVM + score averaging across 5-Fold StratifiedGroupKFold.
    Loops over all specified datasets and subjects.
    """
    if datasets is None:
        datasets = ["BNCI2014_009", "EPFLP300"]
    if subjects is None:
        subjects = range(1, 4)

    all_results = []

    for ds_name in datasets:
        for subj in subjects:
            print(f"\n--- [ Ensemble ] {ds_name} Subject {subj} ---")

            try:
                epochs, X, y = get_clean_data(ds_name, subj)
            except Exception as e:
                print(f"  Skipping subject {subj}: {e}")
                continue

            skf = StratifiedGroupKFold(n_splits=5)
            groups = epochs.metadata['char_id'].values

            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
            ])

            subject_probs   = []
            subject_y       = []
            subject_flashes = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=groups)):
                print(f"  Fold {fold+1}/5...")
                e_tr = epochs[train_idx].copy()
                e_te = epochs[test_idx].copy()
                y_tr = y[train_idx]
                y_te = y[test_idx]

                # AutoReject per fold (ICA already at raw stage)
                e_tr, e_te = run_preprocessing_fold(e_tr, e_te)

                # Decimated feature vectors
                X_tr = extract_p300_features(e_tr.get_data(), decimation_factor=DECIMATION_FACTOR)
                X_te = extract_p300_features(e_te.get_data(), decimation_factor=DECIMATION_FACTOR)

                clf.fit(X_tr, y_tr)
                subject_probs.extend(clf.predict_proba(X_te)[:, 1])
                subject_y.extend(y_te)
                subject_flashes.extend(e_te.metadata['flash_id'].values)

            probs   = np.array(subject_probs)
            y_test  = np.array(subject_y)
            flashes = np.array(subject_flashes)

            char_acc = get_character_prediction(probs, y_test, flashes)
            itr      = get_symbol_itr(36, char_acc, dur=TRIAL_DURATION)

            print(f"  Character Accuracy (N=36): {char_acc*100:.1f}%")
            print(f"  Communication Rate:        {itr:.2f} bits/min")
            all_results.append([ds_name, subj, char_acc, itr])

    print("\n--- ENSEMBLE SUMMARY ---")
    df = pd.DataFrame(all_results, columns=['Dataset', 'Subject', 'Char_Acc', 'ITR_N36'])
    print(df.to_string(index=False))
    df.to_csv('results/ensemble_results.csv', index=False)
    print("Results saved to results/ensemble_results.csv")
    return df


if __name__ == "__main__":
    setup_environment()
    run_ensemble_benchmark()
