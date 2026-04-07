import numpy as np
import mne
from data_loader import get_clean_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

N_CYCLE = 12  # 6 rows + 6 columns per repetition cycle in BNCI2014_009

# Load and preprocess using the centralized A+ Grade pipeline
print("--- Loading data via centralized data_loader ---")
epochs, X_raw, y_raw = get_clean_data(dataset_name='BNCI2014_009', subj=1)

# Flatten epochs for classical classification
X = X_raw.reshape(len(X_raw), -1)
y = y_raw

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train SVM with probability output enabled (required for soft ensemble)
print("--- Training SVM Baseline ---")
clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
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

print("\n--- FINAL ENSEMBLE CONSISTENCY REPORT ---")
print(f"Dataset:                  BNCI2014_009 (Subject 1)")
print(f"Preprocessing:            7-Step Consistent Pipeline (data_loader.py)")
print(f"Block size:               {N_CYCLE} flashes per cycle")
print(f"Single-trial accuracy:    {single_trial_acc:.3f}")
print(f"Ensemble accuracy:        {ensemble_acc:.3f}")
print(f"Improvement:              +{(ensemble_acc - single_trial_acc):.3f}")
