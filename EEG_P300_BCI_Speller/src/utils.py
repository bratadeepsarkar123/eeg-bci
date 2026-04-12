import warnings
import numpy as np
import mne
import config

def setup_environment():
    """Configure MNE logging, suppress warnings, and ensure results dir exists."""
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')
    # Use config.RESULTS_DIR (Path object) to ensure folder is created in the right place
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def get_symbol_itr(n, acc, dur=2.1):
    """
    Compute Information Transfer Rate (ITR) in bits/min.
    n   : number of symbols (N=36 for 6x6 P300 matrix)
    acc : character-level accuracy (0–1)
    dur : trial duration in seconds
    """
    if acc <= 1.0 / n:
        return 0.0
    if acc >= 1.0:
        acc = 0.9999
    bits = (np.log2(n)
            + acc * np.log2(acc)
            + (1 - acc) * np.log2((1 - acc) / (n - 1)))
    return bits * (60.0 / dur)

def get_character_prediction(probs, y_test, flash_ids, char_ids):
    """
    Decode character identity from accumulated flash probabilities.
    Groups flashes by char_ids to ensure resilience against dropped epochs.
    """
    unique_f = np.sort(np.unique(flash_ids))
    if len(unique_f) < 12:
        return 0.0

    rows_ids = unique_f[:6]
    cols_ids = unique_f[6:]
    
    unique_chars = np.unique(char_ids)
    if len(unique_chars) == 0:
        return 0.0

    correct_chars = 0
    for c_id in unique_chars:
        # Select all flashes belonging to this character
        mask = (char_ids == c_id)
        char_probs  = probs[mask]
        char_labels = y_test[mask]
        char_flashes = flash_ids[mask]

        agg_probs  = {}
        target_row = -1
        target_col = -1

        for p, l, f in zip(char_probs, char_labels, char_flashes):
            agg_probs.setdefault(f, []).append(p)
            if l == 1:
                if f in rows_ids: target_row = f
                if f in cols_ids: target_col = f

        # If we have no data for this char, skip
        if not agg_probs:
            continue

        mean_probs = {f: np.mean(v) for f, v in agg_probs.items()}
        
        # Predict row and column by finding the flash with max mean probability
        p_row_vals = [mean_probs.get(r, 0.0) for r in rows_ids]
        p_col_vals = [mean_probs.get(c, 0.0) for c in cols_ids]
        
        pred_row = rows_ids[np.argmax(p_row_vals)]
        pred_col = cols_ids[np.argmax(p_col_vals)]

        if pred_row == target_row and pred_col == target_col:
            correct_chars += 1

    return correct_chars / len(unique_chars)
