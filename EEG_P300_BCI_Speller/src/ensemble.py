import sys
from pathlib import Path

# Path setup for running from any location
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import config
from preprocess import get_clean_data
from engine import run_model_evaluation
from models import get_svm_pipeline
from utils import setup_environment, get_symbol_itr, get_character_prediction

# ---------------------------------------------------------------------------
# NOTE: Character-level decoding (multi-repetition score averaging) is
# implemented in utils.get_character_prediction() and called by BOTH this
# script and the main evaluate.py benchmark. This file is a *standalone*
# runner that isolates the SVM ensemble results in a separate CSV for
# analysis — it does NOT duplicate the decoding logic.
# ---------------------------------------------------------------------------

def run_ensemble_benchmark():
    """
    Standalone benchmark for the SVM Ensemble Speller.

    This complements evaluate.py by providing a dedicated output file
    (ensemble_results.csv) focused exclusively on SVM character accuracy
    and ITR across all configured subjects/datasets.

    The underlying evaluation logic is centralised in engine.run_model_evaluation()
    and utils.get_character_prediction(); no decoding logic is duplicated here.
    """
    print("\n=== P300 Ensemble Speller Benchmark (SVM) ===")
    results_list = []

    clf = get_svm_pipeline()

    for ds_name in config.DATASETS:
        for subj in config.TEST_SUBJECTS:
            print(f"  Crunching ensemble for {ds_name} Subject {subj}...")

            try:
                epochs, X, y = get_clean_data(ds_name, subj)
            except Exception:
                continue

            results = run_model_evaluation(epochs, X, y, clf, "SVM_Ensemble")

            char_acc = get_character_prediction(
                results['probs'], results['true_y'], results['flash_ids']
            )
            itr = get_symbol_itr(36, char_acc, dur=config.TRIAL_DURATION)

            results_list.append({
                'dataset': ds_name,
                'subject': subj,
                'char_acc': char_acc,
                'itr': itr
            })

    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(config.RESULTS_DIR / 'ensemble_results.csv', index=False)
        print(f"\n[DONE] Ensemble results saved to {config.RESULTS_DIR / 'ensemble_results.csv'}")


if __name__ == "__main__":
    setup_environment()
    run_ensemble_benchmark()
