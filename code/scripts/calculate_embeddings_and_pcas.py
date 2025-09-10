from typing import List
import sys
import yaml

with open("parameters.yaml") as f:
    yaml_params = yaml.safe_load(f)
SYS_PATH = yaml_params["SYS_PATH"]
sys.path.append(SYS_PATH)

import pandas as pd
from sklearn.decomposition import PCA
import os




import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")

from src.test_utils import *



RESULTS_PATHS = yaml_params["RESULTS_PATHS"] 
TREATMENT_COLUMN = yaml_params["TREATMENT_COLUMN"] 
TARGET_COLUMN = yaml_params["TARGET_COLUMN_PCA1"]
STDOUT_PATH = yaml_params["STDOUT_PATH"]

sys.stdout = open(STDOUT_PATH, "w", buffering=1)


def main(data_paths:List, treatment_column:str, target_column:str):
    complete_data = pd.DataFrame()
    pcas = {}

    for data_path in data_paths:
        current_rad = data_path.split("/")[-2].split("_")[0]
        current_effect = data_path.split('/')[-1]
        if current_rad not in pcas.keys():
            pcas[current_rad] = {}
        print(current_rad)
        print(current_effect)
        
        print(os.getcwd())

        files = os.listdir(data_path)

        files = [file for file in files if "Embeddings_Meta" in file]
        files = [file.split(".")[0] for file in files]
        file_numbers = [file.split("_")[-1] for file in files]
        comp_data, _,pca_uni = prep_embedding_pca_data_for_tests(data_path, files[0],  treatment_column, target_column)
        comp_data["chunk"] = file_numbers[0]
        comp_data["pca_explained"] = pd.Series(pca_uni.explained_variance_ratio_)
        comp_data["pca_explained"] = comp_data["pca_explained"].ffill()

        if current_effect not in pcas[current_rad].keys():
            pcas[current_rad][current_effect] = {}
        pcas[current_rad][current_effect].update({file_numbers[0]:pca_uni.explained_variance_ratio_})
        for i, f in enumerate(files[1:]):
            nu = i+1
            print(f"Current file number:{file_numbers[nu]}")
            data, _,pca_uni = prep_embedding_pca_data_for_tests(data_path, f"{f}", treatment_column, target_column)
            data["chunk"] = file_numbers[nu]
            data["pca_explained"] = pd.Series(pca_uni.explained_variance_ratio_)
            data["pca_explained"] = data["pca_explained"].ffill()
            comp_data = pd.concat([comp_data,data])
            pcas[current_rad][current_effect].update({file_numbers[nu]:pca_uni.explained_variance_ratio_})
        

        comp_data["scenario"] = f"{data_path.split('/')[-1]}"
        comp_data["radius"] = data_path.split("/")[-2].split("_")[0]
        complete_data = pd.concat([complete_data,comp_data])
        
    pd.to_pickle(complete_data, f"{data_path}/strong_embeddings_radiuses_resized.pkl")
    print("complete_data saved!")
    pd.to_pickle(pcas, f"{data_path}/strong_pca_explained_radiuses_resized.pkl")
    print("pcas explained saved!")

if __name__=="__main__":
    main(
        data_paths = RESULTS_PATHS, 
        treatment_column = TREATMENT_COLUMN, 
        target_column = TARGET_COLUMN
    )
