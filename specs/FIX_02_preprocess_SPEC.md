# SPEC: Fix `02_preprocess.py` — Add Bad Channel Interpolation

## Context
The requirements document (Stage 1, Preprocessing) explicitly lists:
> "Bad channel interpolation (mark bad channels first)"

`02_preprocess.py` has no bad channel handling. This is a 10% rubric item under "Signal processing choices."

## File to Edit
`src/02_preprocess.py`

## Where to Add It
After `raw.set_eeg_reference('average')` and **before** the epoching section.

## Logic to Implement
1. Load the dataset's built-in bad channel list (`raw.info['bads']`).
2. If no bad channels are pre-marked, programmatically identify channels whose standard deviation is more than 3x the median absolute deviation (MAD) across all channels.
3. Mark those channels as bad.
4. Call `raw.interpolate_bads()` to fill them in.

## Complete Replacement File Content

```python
import numpy as np
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

# Stage 1: Signal Cleaning
raw.filter(0.1, 30.0, verbose=False)
raw.notch_filter(50.0, verbose=False)
raw.set_eeg_reference('average', verbose=False)

# Stage 1 (cont): Bad Channel Interpolation
# Detect bad channels using variance thresholding (z-score > 3 across channel stds)
chan_stds = np.std(raw.get_data(), axis=1)
median_std = np.median(chan_stds)
mad = np.median(np.abs(chan_stds - median_std))
z_scores = np.abs(chan_stds - median_std) / (mad + 1e-8)
bad_idx = np.where(z_scores > 3.0)[0]
raw.info['bads'] = [raw.ch_names[i] for i in bad_idx]

if raw.info['bads']:
    print(f"Detected bad channels: {raw.info['bads']}")
    raw.interpolate_bads(reset_bads=True, verbose=False)
else:
    print("No bad channels detected.")

# Stage 2: Epoching — Locked to stimulus onset events
events, _ = mne.events_from_annotations(raw, verbose=False)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0), preload=True, verbose=False)

print("--- Preprocessing Complete ---")
print("Total Epochs Created:", len(epochs))
print("Shape of the mathematical data:", epochs.get_data().shape)
```

## Validation
Run: `python src/02_preprocess.py`  
Expected: Either "No bad channels detected." or a list printed. "Preprocessing Complete" at the end. No errors.
