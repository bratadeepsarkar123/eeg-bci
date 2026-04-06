import numpy as np
import mne
from moabb.datasets import BNCI2014_009
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Load and Filter
ds = BNCI2014_009()
subj = 1
data = ds.get_data(subjects=[subj])
raw = data[subj]['0']['0']
raw.pick_types(eeg=True)
raw.filter(0.1, 30.0, verbose=False)
raw.notch_filter(50.0, verbose=False)       # Remove 50Hz power line noise
raw.set_eeg_reference('average', verbose=False)

# Epoching
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)

# Downsampling by 8
epochs.decimate(8)
features = epochs.get_data().reshape(len(epochs), -1)

print("Feature matrix shape:", features.shape)
print("Labels (0=NonTarget, 1=Target):", dict(zip(*np.unique(epochs.events[:, -1] - 1, return_counts=True))))
