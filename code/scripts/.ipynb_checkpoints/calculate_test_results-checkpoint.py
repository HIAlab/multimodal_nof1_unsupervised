import logging

import sys
sys.path.append(SYS_PATH)

import pandas as pd

from scipy.stats import permutation_test 

from src.test_utils import *

import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")

import yaml

with open("parameters.yaml") as f:
    yaml_params = yaml.safe_load(f)

DATA_PATHS = yaml_params["DATA_PATHS"]
TARGET_PATH = yaml_params["TARGET_PATH"]
GROUP_COLUMN = yaml_params["GROUP_COLUMN"]
TESTS_TO_RUN = yaml_params["TESTS_TO_RUN"]

if SIMULATION_OR_PILOT == "PILOT:
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_PCA1"]
else:
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_AVGSCORE"]

def main(data_paths:str, target_path: str, intervention_column:str, target_column:str,tests_to_run:str):
    for data_path in data_paths:
        
        pkl_name = find_key_name(data_path,target_column)

        logging.info(f"Working on {pkl_name}")
        print(f"On {pkl_name}")
        new_all_pca_vals_radiuses = return_all_vals([data_path], target_column,intervention_column, tests_to_run)
        pd.to_pickle(new_all_pca_vals_radiuses,f"{target_path}/{pkl_name}.pkl")


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, filename="../../../../logs/complex_results.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    main(
        data_paths=DATA_PATHS,
        target_path = TARGET_PATH, 
        target_column=TARGET_COLUMN,
        intervention_column = GROUP_COLUMN,
        tests_to_run=TESTS_TO_RUN,
    )