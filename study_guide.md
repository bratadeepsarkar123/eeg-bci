# BCI Learning Roadmap: From Signal to Speller

Use the keywords and topics below to find in-depth video lectures and tutorials. Each section corresponds to a core module in the **EEG Brain Speller** project.

## 1. Fundamentals of P300 BCI
**Goal:** Understanding the "Brain" part of the project.
- **Search Terms:** "P300 Event Related Potential explained", "Oddball Paradigm EEG", "P300 Speller Row-Column Paradigm".
- **Key Concepts:** 
    - ERP (Event-Related Potential) morphology.
    - The "Target" vs "Non-Target" brain response.
    - Signal-to-Noise Ratio (SNR) challenges in single-trial EEG.

## 2. EEG Signal Processing (The MNE Pipeline)
**Goal:** Understanding how to clean "dirty" brainwaves.
- **Search Terms:** "MNE-Python tutorial preprocessing", "EEG Bandpass and Notch filtering theory", "EEG Re-referencing Average vs Mastoids".
- **Key Concepts:**
    - **ICA (Independent Component Analysis):** Removing eye blinks and EOG artifacts.
    - **Epoching:** Segmenting continuous data into stimulus-locked trials.
    - **Decimation/Downsampling:** Why we reduce the sampling rate for machine learning.

## 3. Classical Machine Learning for BCI
**Goal:** Understanding the LDA and SVM baselines.
- **Search Terms:** "Linear Discriminant Analysis for EEG", "SVM RBF Kernel intuition", "Dealing with class imbalance in BCI".
- **Key Concepts:**
    - **Class Weighting:** Why we penalize "Target" misses more than "Non-Target" misses.
    - **Stratified K-Fold Cross-Validation:** Why we need multiple test-train rotations for scientific validity.

## 4. Deep Learning: The EEGNet Architecture
**Goal:** Understanding the "A-Grade" model from the 2018 Lawhern paper.
- **Search Terms:** "EEGNet architecture explained", "Depthwise and Separable Convolutions intuition", "Temporal vs Spatial filters in CNNs".
- **Key Concepts:**
    - **Temporal Convolution:** Learning frequency filters automatically.
    - **Depthwise Convolution:** Learning spatial patterns (electrode locations).
    - **Separable Convolution:** Reducing parameter count for small datasets.
    - **ELU Activation Function:** Why we use it instead of ReLU for EEG.

## 5. BCI Evaluation & ITR
**Goal:** Proving the system works.
- **Search Terms:** "Information Transfer Rate (ITR) BCI calculation", "Confusion Matrix for imbalanced classification", "BCI competition evaluation metrics".
- **Key Concepts:**
    - **ITR (bits/min):** The gold standard for measuring communication speed.
    - **Target Recall vs Accuracy:** Understanding why 90% accuracy can sometimes be a "fail" in BCI.

## 6. Practical Programming (Libraries)
- **Search Terms:** "MOABB BCI dataset loader tutorial", "PyTorch for EEG classification", "mne.preprocessing.ICA tutorial".
