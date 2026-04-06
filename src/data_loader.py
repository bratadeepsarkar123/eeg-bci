import mne
import numpy as np
from moabb.datasets import BNCI2014_009, EPFLP300
from mne.preprocessing import ICA
import warnings

# Suppress some MNE warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)

SEED = 42

def get_clean_data(dataset_name='BNCI2014_009', subj=1, apply_decimation=True):
    """
    Centralized 7-step P300 Preprocessing Pipeline.
    
    1. Bandpass: 0.1 - 30.0 Hz
    2. Notch: 50.0 Hz
    3. Average Re-referencing
    4. Bad Channel Interpolation (Variance-based)
    5. ICA Artifact Rejection (Excluding first 2 components)
    6. Epoching: tmin=-0.2, tmax=0.8, baseline=(-0.2, 0)
    7. Decimation: Factor of 8
    """
    
    # Step 0: Load Data
    if dataset_name == 'BNCI2014_009':
        ds = BNCI2014_009()
    elif dataset_name == 'EPFLP300':
        ds = EPFLP300()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data = ds.get_data(subjects=[subj])[subj]
    # Pick the first available session and run
    s_key = list(data.keys())[0]
    r_key = list(data[s_key].keys())[0]
    raw = data[s_key][r_key]
    raw.pick_types(eeg=True)
    
    # Preprocessing Step 1 & 2: Filtering
    raw.filter(0.1, 30.0, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)
    
    # Preprocessing Step 3: Average Re-referencing
    raw.set_eeg_reference('average', verbose=False)
    
    # Preprocessing Step 4: Bad Channel Interpolation
    chan_data = raw.get_data()
    chan_stds = np.std(chan_data, axis=1)
    median_std = np.median(chan_stds)
    mad = np.median(np.abs(chan_stds - median_std))
    z_scores = np.abs(chan_stds - median_std) / (mad + 1e-8)
    bad_idx = np.where(z_scores > 3.0)[0]
    raw.info['bads'] = [raw.ch_names[i] for i in bad_idx]
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    # Preprocessing Step 5: ICA Artifact Rejection
    # Using a 1Hz highpass copy for better ICA fit
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = ICA(n_components=min(len(raw.ch_names), 15), random_state=SEED, method='fastica')
    ica.fit(raw_for_ica, verbose=False)
    ica.exclude = [0, 1] 
    ica.apply(raw, verbose=False)
    
    # Preprocessing Step 6: Epoching
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Unified mapping: Target -> 1, Non-Target -> 0
    # BNCI2014_009: Target=2, NonTarget=1
    # EPFLP300: Target=2, NonTarget=1
    # We want labels 0 and 1, but Moabb annotations usually map consistent names.
    # We use event_id names if possible.
    
    target_id = event_id.get('Target')
    nontarget_id = event_id.get('NonTarget')
    
    # If using numerical events, ensure mapping
    epochs = mne.Epochs(raw, events, event_id={'Target': target_id, 'NonTarget': nontarget_id},
                        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, verbose=False)
    
    # Preprocessing Step 7: Decimation
    if apply_decimation:
        epochs.decimate(8)
        
    return epochs, epochs.get_data(), (epochs.events[:, -1] == target_id).astype(int)
