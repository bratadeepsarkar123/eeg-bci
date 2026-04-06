# SPEC: Fix `04_classify.py` — Remove Crash Bug

## Context
`04_classify.py` currently has code at lines 39–55 that references undefined variables (`models`, `x`, `confusion_matrix`). These lines were leftover from an incomplete edit. The file will crash if run. This must be fixed.

## File to Edit
`src/04_classify.py`

## Current Broken Content (lines 36–56)
```python
# 4. Evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    
    # Get predictions using cross-validation
    y_pred = cross_val_predict(model, x, y, cv=cv)
    ...
```

## Required Fix
Delete everything from line 36 onwards. The correct evaluation using `cross_val_predict` is already done in lines 31–34. Replace the entire file with the clean version below.

## Complete Replacement File Content

```python
import numpy as np
import mne
from moabb.datasets import BNCI2014_009
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Data loading and preprocessing
ds = BNCI2014_009()
data = ds.get_data(subjects=[1])
raw = data[1]['0']['0']
raw.pick_types(eeg=True)
raw.filter(0.1, 30.0, verbose=False)
raw.notch_filter(50.0, verbose=False)
raw.set_eeg_reference('average', verbose=False)

# Epoching
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
epochs.decimate(8)

X = epochs.get_data().reshape(len(epochs), -1)
y = epochs.events[:, -1] - 1  # labels: 0=NonTarget, 1=Target

# Models and 5-fold cross-validation
lda = LinearDiscriminantAnalysis()
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [("LDA", lda), ("SVM", svm)]:
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    print(f"\n--- {name} Results ---")
    print(classification_report(y, y_pred, target_names=['NonTarget', 'Target']))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
```

## Validation
Run: `python src/04_classify.py`  
Expected: No errors. Two classification reports printed (LDA and SVM).
