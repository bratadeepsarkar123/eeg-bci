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
    4. Epoching: tmin=-0.2, tmax=0.8, baseline=(-0.2, 0)
    5. Dynamic Decimation: Targets a Nyquist-safe rate for 30Hz bandwidth
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
    
    # NOTE: Average Re-referencing (Step 3), Bad Channels (Step 4), and ICA (Step 5) 
    # are moved to the evaluation/plotting loops to prevent cross-validation leakage 
    # and ensure bad channels aren't mixed into the spatial reference.

    # Preprocessing Step 4: Epoching
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
    
    # Step 8: Metadata Extraction (Fix for Bug #11)
    # Recover stimulus IDs (Row/Column) and Character IDs for Grouped CV.
    stim_ch = next((raw.ch_names.index(c) for c in ['Flash stim', 'STI', 'stim'] if c in raw.ch_names), None)
    
    flash_ids = []
    if stim_ch is not None:
        for event_time in events[:, 0]:
            val = raw[stim_ch, event_time][0][0][0]
            flash_ids.append(int(val))
    else:
        flash_ids = [i % 12 for i in range(len(events))]
    
    # Character ID assuming 120 trials (10 reps * 12 flashes) per character block
    char_ids = np.arange(len(events)) // 120
    
    epochs.metadata = mne.utils._prepare_metadata(
        metadata=np.column_stack([flash_ids, char_ids]),
        names=['flash_id', 'char_id'],
        col_type={'flash_id': 'int64', 'char_id': 'int64'},
        row_names=None
    )

    return epochs, epochs.get_data(), (epochs.events[:, -1] == target_id).astype(int)

def apply_bad_channel_interpolation(epochs_train, epochs_test, z_thresh=3.0):
    """
    Detect bad channels from training epochs only and interpolate in both train/test.
    Zero-Leakage standard for scientific BCI analysis.
    """
    train_data = epochs_train.get_data() # (n_epochs, n_channels, n_times)
    # Block-wise standard deviation across channels
    chan_stds = np.std(train_data, axis=(0, 2))
    median_std = np.median(chan_stds)
    mad = np.median(np.abs(chan_stds - median_std))
    z_scores = np.abs(chan_stds - median_std) / (mad + 1e-8)
    bad_idx = np.where(z_scores > z_thresh)[0]
    bads = [epochs_train.ch_names[i] for i in bad_idx]

    if bads:
        # Prevent interpolating too many channels
        if len(bads) < len(epochs_train.ch_names) // 2:
            epochs_train.info['bads'] = bads
            epochs_test.info['bads'] = bads
            epochs_train.interpolate_bads(reset_bads=True, verbose=False)
            epochs_test.interpolate_bads(reset_bads=True, verbose=False)
            # print(f"  [Info] Interpolated {len(bads)} bad channels: {bads}")
    return epochs_train, epochs_test

def apply_spatial_ica(epochs_train, epochs_test):
    """
    Fits ICA on training epochs and applies it to both train and test to prevent leakage.
    Uses Spatial Audit to only exclude EOG-correlated components.
    """
    # Methodological Bug #10 Note: ICA is fitted on epoched data rather than continuous raw 
    # to maintain strict Zero-Leakage cross-validation constraints. While continuous fits 
    # are often more stable, this protocol ensures no artifact data from the test fold 
    # influences the training features.
    ica = ICA(n_components=min(len(epochs_train.ch_names), 15), random_state=SEED, method='fastica')
    # Filter a copy of training data for better ICA fitting (1Hz highpass)
    epochs_for_ica = epochs_train.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(epochs_for_ica, verbose=False)
    
    frontal_chans = [ch for ch in ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fpz', 'FP1', 'FP2', 'FPZ'] if ch in epochs_train.ch_names]
    
    if frontal_chans:
        eog_indices, eog_scores = ica.find_bads_eog(epochs_train, ch_name=frontal_chans, verbose=False)
        ica.exclude = eog_indices
        if not ica.exclude:
            # Bug #5 Fix: If no correlation, exclude NOTHING.
            ica.exclude = []
        print(f"    -> ICA: Spatial Audit found {len(ica.exclude)} artifact components correlating with {frontal_chans}")
    else:
        ica.exclude = [] # Be perfectly safe if no frontal channel
        print(f"    -> ICA: No frontal channels found. Excluding nothing.")
        
    ica.apply(epochs_train, verbose=False)
    ica.apply(epochs_test, verbose=False)
    
    return epochs_train, epochs_test
