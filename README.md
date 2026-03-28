# EEG Brain Speller Project

This project implements an EEG-based Brain-Computer Interface (BCI) speller system that allows character typing using brain signals. It uses the P300 paradigm and detects specific brainwave patterns in response to visual stimuli.

## Project Structure
- `data/`: Raw and preprocessed EEG files
- `notebooks/`: Jupyter notebooks for data exploration
- `src/`: Core Python source code for signal processing and classification
- `results/`: Saved models, plots, and evaluation metrics

## Project Results (Final - 5-Fold Averaged)

| Model | Accuracy | Target Recall | Information Transfer Rate (ITR) |
| :--- | :--- | :--- | :--- |
| **LDA (Baseline)** | 68.9% | 48.9% | 80.49 bits/min |
| **SVM (Weighted)** | 87.2% | 40.5% | **118.73 bits/min** |
| **EEGNet (Deep Learning)** | **86.8%** | **79.1%** | **117.91 bits/min** |

> [!NOTE]
> The results above are averaged across **5-Fold Stratified Cross-Validation** to ensure scientific robustness. Artifact rejection (ICA), 50Hz notch filtering, and average re-referencing have been applied according to the project rubric.

### Next Steps for Submission
- View all performance charts in the `results/` directory.
- Review `research_log.md` for the technical development narrative.
- Use `src/06_final_evaluation.py` to re-generate these results locally.

## Environment Setup
The project uses a Python virtual environment:
```bash
# Recreate the environment
python -m venv eeg_env

# Activate the virtual environment
source eeg_env/bin/activate  # Linux/macOS
eeg_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
