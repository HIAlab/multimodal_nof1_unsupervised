import sys
sys.path.append(SYS_PATH)

import pandas as pd
from sklearn.decomposition import PCA
import os




import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")

from src.test_utils import *

import yaml

with open("parameters.yaml") as f:
    yaml_params = yaml.safe_load(f)

DATA_PATHS = yaml_params["DATA_PATHS"] 
TREATMENT_COLUMN = yaml_params["TREATMENT_COLUMN"] 

if SIMULATION_OR_PILOT == "PILOT:
    TARGET_COLUMN = TARGET_COLUMN_PCA1
else:
    TARGET_COLUMN = TARGET_COLUMN_AVGSCORE


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
        files = os.listdir(data_path)

        files = [file for file in files if "Embeddings_Meta" in file]
        files = [file.split(".")[0] for file in files]
        file_numbers = [file.split("_")[-1] for file in files]
        comp_data, _,pca_uni = prep_embedding_pca_data_for_tests(data_path, files[0])
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
        
    pd.to_pickle(complete_data, "strong_embeddings_radiuses_resized.pkl")
    print("complete_data saved!")
    pd.to_pickle(pcas, "strong_pca_explained_radiuses_resized.pkl")
    print("pcas explained saved!")

if __name__=="__main__":
    main(
        data_paths = DATA_PATHS, 
        treatment_column = TREATMENT_COLUMN, 
        target_column = TARGET_COLUMN
    )
