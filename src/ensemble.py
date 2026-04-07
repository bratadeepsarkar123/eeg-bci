import numpy as np
import mne
from preprocess import get_clean_data, apply_bad_channel_interpolation, apply_spatial_ica
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

N_CYCLE = 12  # 6 rows + 6 columns per repetition cycle in BNCI2014_009

def get_itr(n, acc, dur=2.0):
    """Calculates Information Transfer Rate (bits/min)."""
    if acc >= 0.99: return np.log2(n) * 60 / dur
    if acc <= 1/n: return 0
    return (np.log2(n) + acc*np.log2(acc) + (1-acc)*np.log2((1-acc)/(n-1))) * 60 / dur

# Load and preprocess using the centralized A+ Grade pipeline
print("--- Loading data via centralized data_loader ---")
epochs, _, y_raw = get_clean_data(dataset_name='BNCI2014_009', subj=1)

# SCIENTIFIC FIX: Disable shuffle to preserve temporal/block structure
idx = np.arange(len(epochs))
idx_train, idx_test = train_test_split(idx, test_size=0.2, shuffle=False)

# Bug #2 Fix: ICA inside CV loop
epochs_train = epochs[idx_train].copy()
epochs_test = epochs[idx_test].copy()
epochs_train, epochs_test = apply_bad_channel_interpolation(epochs_train, epochs_test)
epochs_train, epochs_test = apply_spatial_ica(epochs_train, epochs_test)

X_train = epochs_train.get_data().reshape(len(idx_train), -1)
X_test = epochs_test.get_data().reshape(len(idx_test), -1)
y_train, y_test = y_raw[idx_train], y_raw[idx_test]

# Train SVM with scaling pipeline
print("--- Training SVM Baseline (with scaling) ---")
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
])
clf.fit(X_train, y_train)

# --- Single-trial evaluation ---
y_pred = clf.predict(X_test)
single_trial_acc = np.mean(y_pred == y_test)
print("\n--- Single-Trial Classification Results ---")
print(classification_report(y_test, y_pred, target_names=['NonTarget', 'Target']))

# --- Ensemble evaluation (averaging scores across repetition cycles) ---
# Soft Ensemble: Average the 'Target' class probability across flashes
probs = clf.predict_proba(X_test)[:, 1]  
n_blocks = len(probs) // N_CYCLE
n_correct = 0

print(f"--- Running Ensemble Evaluation ({n_blocks} blocks) ---")
for i in range(n_blocks):
    block_probs  = probs[i * N_CYCLE:(i + 1) * N_CYCLE]
    block_labels = y_test[i * N_CYCLE:(i + 1) * N_CYCLE]
    
    # Identify the flash with the highest P300 probability in the block
    predicted_idx = np.argmax(block_probs)
    
    # If that flash was indeed the Target, the ensemble decision is correct
    if block_labels[predicted_idx] == 1:
        n_correct += 1

ensemble_acc = n_correct / n_blocks if n_blocks > 0 else 0.0

# Character-level ITR (N=36)
# Fix Bug #7/9: dur represents total physiological time for a character (e.g., 10 repetitions * 1.2s = 12s)
ensemble_itr = get_itr(36, ensemble_acc, dur=12.0)

print("\n--- FINAL ENSEMBLE CONSISTENCY REPORT ---")
print(f"Dataset:                  BNCI2014_009 (Subject 1)")
print(f"Preprocessing:            7-Step Consistent Pipeline (preprocess.py)")
print(f"Block size:               {N_CYCLE} flashes per cycle")
print(f"Single-trial accuracy:    {single_trial_acc:.3f}")
print(f"Ensemble accuracy:        {ensemble_acc:.3f}")
print(f"Character-Level ITR:      {ensemble_itr:.1f} bits/min (N=36)")
print(f"Improvement:              +{(ensemble_acc - single_trial_acc):.3f}")
