# SPEC: New File `07_erp_plot.py` — ERP Waveform Visualisation

## Context
The requirements document (Section 2.2, Libraries) states:
> "matplotlib ≥3.7 — Plotting ERP waveforms and results"

No ERP waveform plot exists in the codebase. This is a key visual for the project report and a direct requirement. An ERP plot shows the **mean time-series signal** of Target vs Non-Target epochs across all channels, clearly visualising the P300 component (~300–500ms post-stimulus).

## New File to Create
`src/07_erp_plot.py`

## Exact Logic to Implement

### Step 1: Load and preprocess data (same pipeline as 02_preprocess.py)
- Use BNCI2014_009, subject 1.
- Apply: bandpass (0.1–30Hz), notch (50Hz), average reference.
- Epoch: tmin=-0.2, tmax=0.8, baseline=(-0.2, 0).
- Do NOT decimate — keep full time resolution for a smooth ERP curve.
- Labels: `epochs.events[:, -1] - 1` → 0=NonTarget, 1=Target.

### Step 2: Separate epochs by label
```python
target_epochs = epochs[epochs.events[:, -1] - 1 == 1]
nontarget_epochs = epochs[epochs.events[:, -1] - 1 == 0]
```

### Step 3: Average across all channels and all trials
- Compute mean over trials for each class, then mean over channels.
- `target_erp = target_epochs.get_data().mean(axis=0).mean(axis=0)` → shape: (n_times,)
- `nontarget_erp = nontarget_epochs.get_data().mean(axis=0).mean(axis=0)` → shape: (n_times,)

### Step 4: Get the time axis
```python
times = epochs.times * 1000  # Convert to milliseconds
```

### Step 5: Plot
- Create a single figure with:
    - X-axis: Time (ms), range -200ms to 800ms.
    - Y-axis: Amplitude (µV) — multiply `get_data()` values by `1e6`.
    - Plot Target ERP as a **solid blue line**, label="Target (P300 present)".
    - Plot NonTarget ERP as a **dashed red line**, label="Non-Target".
    - Draw a vertical dashed grey line at x=0 (stimulus onset).
    - Draw a horizontal dashed grey line at y=0.
    - Add a grey shaded band from x=250 to x=500 to highlight the P300 window.
    - Title: `"Grand Average ERP — Target vs Non-Target (P300 Speller)"`
    - X-axis label: `"Time (ms)"`
    - Y-axis label: `"Amplitude (µV)"`
    - Add `plt.legend()`.
    - Add `plt.grid(True, alpha=0.3)`.
    - Save figure to `results/erp_waveform.png` at `dpi=200`.
    - Print: `"ERP plot saved to results/erp_waveform.png"`.

## Complete File Content

```python
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
```

## Validation
Run: `python src/07_erp_plot.py`  
Expected: `results/erp_waveform.png` is created. The plot visually shows a positive peak ("bump") in the Target ERP between 250–500ms that is absent or much smaller in the Non-Target ERP.
