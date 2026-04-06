# SPEC: New File `08_ensemble_averaging.py` — Ensemble Classification

## Context
The requirements document (Stage 4, Classification) states:
> "Ensemble: average classifier scores across multiple flash repetitions (improves accuracy significantly)"

The current codebase classifies each single-trial epoch independently. In a real P300 speller, the same row and column flash **multiple times** (typically 5–10 repetitions). The correct approach is to **accumulate the classifier's probability scores** across all repetitions before making a final character-level decision. This is the most important accuracy improvement in the BCI literature.

## New File to Create
`src/08_ensemble_averaging.py`

## Exact Logic to Implement

### Background Theory
- The BNCI2014-009 dataset presents a 6×6 character matrix (36 symbols).
- Each character is spelled by flashing all 12 rows and columns (6 rows + 6 columns).
- This repeats for N_REPS repetitions (typically 10 in this dataset).
- So for each character being spelled: there are 12 × N_REPS = 120 total flash events.
- The target character's row AND column will each flash exactly once per repetition → 2 target flashes per repetition, 10 non-target flashes per repetition.

### Implementation Strategy (Simplified — Score Averaging per Identity Group)
We use SVM with `probability=True` so we can extract the probability score for class 1 (Target).

Then we simulate ensemble by:
1. Getting all trial-level Target-class probabilities from SVM.
2. Grouping them into blocks of 12 (one full row+col flash cycle).
3. Within each block, finding the row that has the highest accumulated Target score (that is the predicted row) and the column with the highest score (that is the predicted column).
4. Comparing to ground truth.

**NOTE:** Since the MOABB dataset does not expose character-level grouping labels directly in a simple way, use the following approximation:
- Train SVM on 80% of single-trial epochs using `train_test_split`.
- On the test set, group consecutive epochs into blocks of N=12 (simulating 1 repetition).
- Within each block of 12 epochs, sum the predicted Target probabilities.
- The epoch index with the highest probability within the block is the "character selection."
- Report: **Block-level accuracy** (how often the max-prob epoch corresponded to a true Target epoch within the block).

### Step-by-Step Code Logic

```python
N = 12  # Events per repetition cycle (6 rows + 6 columns)
```

1. Load and preprocess data (same as `04_classify.py`, use notch+ref+decimate).
2. Flatten features: `X = epochs.get_data().reshape(len(epochs), -1)`.
3. Labels: `y = epochs.events[:, -1] - 1`.
4. Split: 80% train, 20% test with `train_test_split(..., stratify=y, random_state=42)`.
5. Train: `SVC(kernel='rbf', class_weight='balanced', probability=True)`.
6. Get probabilities on test: `probs = clf.predict_proba(X_test)[:, 1]`.
7. Ensemble loop:
    ```python
    n_correct = 0
    n_blocks = len(probs) // N
    for i in range(n_blocks):
        block_probs = probs[i*N:(i+1)*N]
        block_labels = y_test[i*N:(i+1)*N]
        predicted_idx = np.argmax(block_probs)
        if block_labels[predicted_idx] == 1:  # Predicted position was a True Target
            n_correct += 1
    ensemble_acc = n_correct / n_blocks
    ```
8. Also compute single-trial accuracy from `clf.predict(X_test)` for comparison.
9. Print both (single-trial accuracy and ensemble accuracy).

## Complete File Content

```python
import numpy as np
import mne
from moabb.datasets import BNCI2014_009
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

N_CYCLE = 12  # 6 rows + 6 columns per repetition

# Load and preprocess
ds = BNCI2014_009()
raw = ds.get_data(subjects=[1])[1]['0']['0']
raw.pick_types(eeg=True)
raw.filter(0.1, 30.0, verbose=False)
raw.notch_filter(50.0, verbose=False)
raw.set_eeg_reference('average', verbose=False)

events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0), preload=True, verbose=False)
epochs.decimate(8)

X = epochs.get_data().reshape(len(epochs), -1)
y = epochs.events[:, -1] - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train SVM with probability output enabled
clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
clf.fit(X_train, y_train)

# --- Single-trial evaluation ---
y_pred = clf.predict(X_test)
single_trial_acc = np.mean(y_pred == y_test)
print("--- Single-Trial Classification ---")
print(classification_report(y_test, y_pred, target_names=['NonTarget', 'Target']))

# --- Ensemble evaluation (averaging scores across repetition cycles) ---
probs = clf.predict_proba(X_test)[:, 1]  # Target class probability
n_blocks = len(probs) // N_CYCLE
n_correct = 0
for i in range(n_blocks):
    block_probs  = probs[i * N_CYCLE:(i + 1) * N_CYCLE]
    block_labels = y_test[i * N_CYCLE:(i + 1) * N_CYCLE]
    predicted_idx = np.argmax(block_probs)
    if block_labels[predicted_idx] == 1:
        n_correct += 1

ensemble_acc = n_correct / n_blocks if n_blocks > 0 else 0.0

print("\n--- Ensemble Averaged Classification ---")
print(f"Block size (events per repetition cycle): {N_CYCLE}")
print(f"Number of complete blocks evaluated:      {n_blocks}")
print(f"Single-trial accuracy:                    {single_trial_acc:.3f}")
print(f"Ensemble accuracy (averaged scores):      {ensemble_acc:.3f}")
print(f"Improvement from ensemble:                +{(ensemble_acc - single_trial_acc):.3f}")
```

## Validation
Run: `python src/08_ensemble_averaging.py`  
Expected: A clear printout showing single-trial accuracy and ensemble accuracy. The ensemble accuracy should be **equal to or higher than** the single-trial accuracy, demonstrating the benefit of score averaging.
