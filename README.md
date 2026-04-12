# Winter-projects-25-26

This repository contains academic project submissions for the Winter Projects 2025–26 programme.

---

## Projects Included

### EEG P300 BCI Speller — Bratadeep Sarkar

**Directory**: [`EEG_P300_BCI_Speller/`](./EEG_P300_BCI_Speller/)

A full end-to-end **P300 Brain-Computer Interface Speller** pipeline implementing rigorous academic-grade signal processing, classical ML baselines, and deep learning classification.

**Key Features:**
- Bandpass (0.1–30 Hz), Notch (50 Hz), Avg-Reference, Bad-Channel Interpolation, and ICA artifact removal
- Xdawn spatial filtering and Riemannian covariance geometry feature extraction
- LDA, SVM (RBF), EEGNet (PyTorch), and MDM (Riemannian) classifiers
- Zero-leakage 5-fold Group Cross-Validation (stratified by character block)
- Multi-subject benchmarking across `BNCI2014_009` and `EPFLP300` datasets
- ITR reporting for 36-symbol speller under 10-repetition protocol

**Stack**: Python 3.10+, MNE-Python, MOABB, PyTorch, scikit-learn, pyRiemann

**How to run:**
```bash
cd EEG_P300_BCI_Speller
pip install -r requirements.txt
python src/evaluate.py
```

See [`EEG_P300_BCI_Speller/README.md`](./EEG_P300_BCI_Speller/README.md) for detailed documentation, results, and compliance audit.

---
