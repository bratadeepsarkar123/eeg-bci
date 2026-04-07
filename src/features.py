import numpy as np

def extract_p300_features(epochs_data, decimation_factor=1):
    """
    Standard P300 feature extraction by downsampling the epoch waveform.
    Note: Downsampling is integrated into the preprocessing pipeline for efficiency.
    """
    if decimation_factor > 1:
        return epochs_data[:, :, ::decimation_factor].reshape(len(epochs_data), -1)
    return epochs_data.reshape(len(epochs_data), -1)

# Note: For our advanced EEGNet, the raw epoched data 
# is fed directly into the spatial-temporal convolutions.
