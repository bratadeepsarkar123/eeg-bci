import mne
from moabb.datasets import BNCI2014_009
import warnings

warnings.filterwarnings('ignore')

# Dataset setup
ds = BNCI2014_009()
subj = 1
data = ds.get_data(subjects=[subj])

# Extracting first session and run
raw = data[subj]['0']['0']
raw.pick_types(eeg=True, stim=False)

print(f"Channels: {raw.info['nchan']}")
print(f"Sampling Rate: {raw.info['sfreq']} Hz")
print(f"Data shape: {raw.get_data().shape}")
print("Channels:", raw.info['ch_names'])

events, ids = mne.events_from_annotations(raw)
print("\n--- Event Info ---")
print("Total events:", len(events))
print("Event mapping:", ids)
raw.plot()