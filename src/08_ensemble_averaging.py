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
