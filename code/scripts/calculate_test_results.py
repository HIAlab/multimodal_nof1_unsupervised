import logging

import sys


import pandas as pd

from scipy.stats import permutation_test 

from src.test_utils import *

import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")

import yaml

with open("parameters.yaml") as f:
    yaml_params = yaml.safe_load(f)

SYS_PATH = yaml_params["SYS_PATH"]

sys.path.append(SYS_PATH)

RESULTS_PATHS = yaml_params["RESULTS_PATHS"]
if isinstance(RESULTS_PATHS, str):
    RESULTS_PATHS = eval(RESULTS_PATHS)

GROUP_COLUMN = yaml_params["GROUP_COLUMN"]
TESTS_TO_RUN = yaml_params["TESTS_TO_RUN"]
TARGET_COLUMN = yaml_params["TARGET_COLUMN_PCA1"]

STDOUT_PATH = yaml_params["STDOUT_PATH"]
sys.stdout = open(STDOUT_PATH, "w", buffering=1)

def main(data_paths:str, intervention_column:str, target_column:str,tests_to_run:str):
    for data_path in data_paths:
        
        pkl_name = find_key_name(data_path,target_column)

        logging.info(f"Working on {pkl_name}")
        print(f"On {pkl_name}")
        new_all_pca_vals_radiuses = return_all_vals([data_path], target_column,intervention_column, tests_to_run)
        pd.to_pickle(new_all_pca_vals_radiuses,f"{data_path}/{pkl_name}.pkl")


if __name__=="__main__":
    main(
        data_paths=RESULTS_PATHS, 
        target_column=TARGET_COLUMN,
        intervention_column = GROUP_COLUMN,
        tests_to_run=TESTS_TO_RUN,
    )