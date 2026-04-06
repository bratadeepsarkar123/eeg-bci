import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from moabb.datasets import BNCI2014_009
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

os.makedirs('results', exist_ok=True)

# Load and preprocess
ds = BNCI2014_009()
raw = ds.get_data(subjects=[1])[1]['0']['0']
raw.pick_types(eeg=True)
raw.filter(0.1, 30.0, verbose=False)
raw.notch_filter(50.0, verbose=False)
raw.set_eeg_reference('average', verbose=False)

# Epoch — full resolution (no decimation, for smooth ERP curve)
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0), preload=True, verbose=False)

labels = epochs.events[:, -1] - 1
times = epochs.times * 1000  # milliseconds

# Separate by class and compute grand average (mean over trials then channels)
target_erp    = epochs.get_data()[labels == 1].mean(axis=0).mean(axis=0) * 1e6
nontarget_erp = epochs.get_data()[labels == 0].mean(axis=0).mean(axis=0) * 1e6

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, target_erp, color='steelblue', linewidth=2, label='Target (P300 present)')
ax.plot(times, nontarget_erp, color='tomato', linewidth=2, linestyle='--', label='Non-Target')
ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, label='Stimulus onset')
ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
ax.axvspan(250, 500, alpha=0.12, color='steelblue', label='P300 window (250–500ms)')
ax.set_title('Grand Average ERP — Target vs Non-Target (P300 Speller)', fontsize=13)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/erp_waveform.png', dpi=200)
plt.close()
print("ERP plot saved to results/erp_waveform.png")
