import mne
from moabb.datasets import BNCI2014_009
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Load raw file
ds = BNCI2014_009()
subj = 1
data = ds.get_data(subjects=[subj])
raw = data[subj]['0']['0']
raw.pick_types(eeg=True)

# Signal cleaning
raw.filter(0.1, 30.0)
raw.notch_filter(50.0)
raw.set_eeg_reference('average')

# Epoching - Locking to stimuli
events, _ = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, 
                    baseline=(-0.2, 0), preload=True)

print("Epochs shape:", epochs.get_data().shape)
print("Event counts:", epochs.event_id)

print("--- Preprocessing Complete ---")
print("Total Epochs Created:", len(epochs))
print("Shape of the mathematical data:", epochs.get_data().shape)
