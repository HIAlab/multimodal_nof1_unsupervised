import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate test results from embedding csv files.")
    parser.add_argument("--config", default=None, help="pilot, sim, or path to a config yaml")
    return parser.parse_args()


ARGS = parse_args()
CONFIG = load_config(ARGS.config)

STDOUT_PATH = resolve_path(CONFIG["STDOUT_PATH"], CONFIG)
Path(STDOUT_PATH).parent.mkdir(parents=True, exist_ok=True)
sys.stdout = open(STDOUT_PATH, "w", buffering=1)

PREPATH = resolve_path(CONFIG["EMBEDDINGS_PATH"], CONFIG)
RESULTS_PATH = resolve_path(CONFIG["RESULTS_PATH"], CONFIG)
TREATMENT_COLUMN = CONFIG["TREATMENT_COLUMN"]
TESTS_TO_RUN = CONFIG["TESTS_TO_RUN"]
TARGET_COLUMN = CONFIG["TARGET_COLUMN_PCA1"]
ALPHA = CONFIG["ALPHA"]
RADIUS = CONFIG["RADIUS"]
SCENARIOS = CONFIG["SCENARIOS"]
SIM = CONFIG["SIM"]
COLS_TO_GROUPBY = CONFIG["COLS_TO_GROUPBY"]

from src.clean_test_utils_consistent import calculate_tests_from_embeddings


def main():
    print(f"SIM {SIM}")
    results = calculate_tests_from_embeddings(
        prepath=PREPATH,
        radius=RADIUS,
        scenarios=SCENARIOS,
        treatment_column=TREATMENT_COLUMN,
        target_column=TARGET_COLUMN,
        tests_to_run=TESTS_TO_RUN,
        cols_to_groupby=COLS_TO_GROUPBY,
        alpha_level=ALPHA,
        sim=SIM,
    )
    results_dir = Path(RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = "pilot" if not SIM else "sim"
    pd.to_pickle(results, results_dir / f"geordnet_calculated_test_results_{mode}.pkl")


if __name__ == "__main__":
    main()
