import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import warnings
from data_loader import get_clean_data

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
os.makedirs('results', exist_ok=True)

def plot_dataset_erp(ax, dataset_name, subj=1):
    """Plots Target vs Non-Target ERP for a specific dataset/subject."""
    epochs_obj, X, y = get_clean_data(dataset_name=dataset_name, subj=subj, apply_decimation=False)
    
    # Calculate Grand Average (mean trials, then mean channels)
    # X shape is (n_epochs, n_channels, n_times)
    target_erp = X[y == 1].mean(axis=0).mean(axis=0) * 1e6
    nontarget_erp = X[y == 0].mean(axis=0).mean(axis=0) * 1e6
    times = epochs_obj.times * 1000
    
    ax.plot(times, target_erp, color='#1f77b4', linewidth=2.5, label='Target (P300)')
    ax.plot(times, nontarget_erp, color='#ff7f0e', linewidth=2.0, linestyle='--', label='Non-Target')
    
    # Styling
    ax.axvline(0, color='black', alpha=0.3, label='Onset')
    ax.axvspan(250, 500, alpha=0.15, color='#1f77b4', label='P300 Zone')
    ax.set_title(f'ERP: {dataset_name} (Subj {subj})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize='small')

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    print("--- Generating Comparative ERP Plots ---")
    
    print("  - Plotting BNCI2014_009...")
    plot_dataset_erp(ax1, 'BNCI2014_009', subj=1)
    
    print("  - Plotting EPFLP300...")
    plot_dataset_erp(ax2, 'EPFLP300', subj=1)
    
    plt.suptitle('Multi-Dataset ERP Comparison (Target vs Non-Target)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/comparative_erp.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    print("Success: Comparative ERP plot saved to results/comparative_erp.png")
