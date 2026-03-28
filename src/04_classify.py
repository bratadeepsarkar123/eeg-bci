import numpy as np
import mne
from moabb.datasets import BNCI2014_009
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# Data loading
objs = BNCI2014_009()
data = objs.get_data(subjects=[1])
raw = data[1]['0']['0']
raw.filter(0.1, 30.0, verbose=False)

# Epoching
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
epochs.decimate(8)

X = epochs.get_data().reshape(len(epochs), -1)
y = epochs.events[:, -1]

# Models
lda = LinearDiscriminantAnalysis()
svm = SVC(kernel='rbf', class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [("LDA", lda), ("SVM", svm)]:
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    print(f"\n--- {name} Results ---")
    print(classification_report(y, y_pred))

# 4. Evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    
    # Get predictions using cross-validation
    y_pred = cross_val_predict(model, x, y, cv=cv)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    
    # Print Metrics (Stage 5 requirements)
    print(classification_report(y, y_pred, target_names=['NonTarget', 'Target']))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    # Calculate ITR (36 symbols, ~2s per trial)
    itr = calculate_itr(36, accuracy, 2.0)
    print(f"ITR: {itr:.2f} bits/minute")
