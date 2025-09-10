from datetime import datetime
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip,RandomResizedCrop, RandomRotation, ColorJitter, RandomCrop, RandomEqualize, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from src.AE import AutoEncoder
from src.EmbeddingsAE import AnalyzeStudyAE
from src.AE_utils import default_trans, non_trans
import yaml
import sys
import os

with open("parameters.yaml") as f:
    yaml_params = yaml.safe_load(f)

STDOUT_PATH = yaml_params["STDOUT_PATH"]

sys.stdout = open(STDOUT_PATH, "w", buffering=1)
DATA_DIR_PATH = yaml_params["DATA_DIR_PATH"]
OUTPUT_DIR_PATH = yaml_params["OUTPUT_DIR_PATH"]
SPECIFIC_DIRS = yaml_params["SPECIFIC_DIRS"]
if isinstance(SPECIFIC_DIRS, str):
    FINAL_IMAGE_SIZE = eval(SPECIFIC_DIRS)

FINAL_IMAGE_SIZE = yaml_params["FINAL_IMAGE_SIZE"]
N_EPOCHS = yaml_params["N_EPOCHS"]
BATCH_SIZE = yaml_params["BATCH_SIZE"]
LEARNING_RATE = yaml_params["LEARNING_RATE"]

ALL_IDS = yaml_params["ALL_IDS"] 
if isinstance(ALL_IDS, str) and ALL_IDS.startswith("range(") and ALL_IDS.endswith(")"):
    try:
        args = [int(x.strip()) for x in ALL_IDS[6:-1].split(",")]
        ALL_IDS = list(range(*args))
    except Exception as e:
        raise ValueError(f"Invalid range specification: {ALL_IDS}") from e


print(ALL_IDS[:10])
MAX_CHUNK_SIZE = yaml_params["MAX_CHUNK_SIZE"] 
FIRST_CHUNK = yaml_params["FIRST_CHUNK"]
LAST_CHUNK = yaml_params["LAST_CHUNK"]
RECREATE = yaml_params["RECREATE"]

TREATMENT_COLUMN = yaml_params["TREATMENT_COLUMN"]
SIMULATION_OR_PILOT = yaml_params["SIMULATION_OR_PILOT"]

if SIMULATION_OR_PILOT == "PILOT":
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_AVGSCORE"]
    CSV_FILE = yaml_params["CSV_FILE_PILOT"]
else:
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_PCA1"]
    CSV_FILE = yaml_params["CSV_FILE_SIMULATION"]

# def print(text):
#     builtins.print(text)
#     os.fsync(sys.stdout)

def fast_scandir(dirname, specific_dirs=[]):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    
    print(f"ALL  SUBFOLDERS:{subfolders}")
    subsubfolders = []
    for subdirname in list(subfolders):
        subsubfolders.extend([f.path for f in os.scandir(subdirname) if f.is_dir()])
    print(f"ALL  SUB SUBFOLDERS:{subsubfolders}")
    all_specific_subfolders = []
    if specific_dirs:
        for pair in specific_dirs:
            subsubsubfolders = subsubfolders.copy()
            for i in pair:
                subsubsubfolders = [s for s in subsubsubfolders if i in s]
            all_specific_subfolders.extend(subsubsubfolders)
    else:
        all_specific_subfolders = subsubfolders

    print(f"ALL SPECIFIC SUBFOLDERS:{all_specific_subfolders}")

    all_specific_subfolders = [s for s in all_specific_subfolders if ".ipynb_checkpoints" not in s]
    return all_specific_subfolders

import os

def collect_image_dirs(data_dir_path, specific_dirs=None):
    """
    Collect image directory paths depending on structure and specific_dirs.
    
    Parameters
    ----------
    data_dir_path : str
        Path to the main dataset folder.
    specific_dirs : list or None
        If list is empty: behavior depends on dataset type.
        If list is non-empty: collect only those subfolders.
    
    Returns
    -------
    list of str
        List of directory paths.
    """
    specific_dirs = specific_dirs or []

    dataset_name = os.path.basename(data_dir_path.rstrip("/"))

    # Case 1: Acne_Nof1_trial
    if dataset_name == "Acne_Nof1_trial":
        if not specific_dirs:
            return [data_dir_path]
        else:
            return [os.path.join(data_dir_path, str(d)) for d in specific_dirs]

    # Case 2: Simulation
    elif dataset_name == "Simulation":
        if not specific_dirs:
            return [
                os.path.join(data_dir_path, d)
                for d in os.listdir(data_dir_path)
                if os.path.isdir(os.path.join(data_dir_path, d))
            ]
        else:
            return [os.path.join(data_dir_path, str(d)) for d in specific_dirs]

    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")


def main(data_dir_path:str, output_dir_path:str, all_IDs: list, 
         max_chunk_size:int, first_chunk:int, last_chunk:int, specific_dirs:list, recreate=RECREATE, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE): 
    """Main funtion to create embeddings

    Args:
        data_dir_path (str): Path to data.
        output_dir_path (str): Path for the results.
        n_epochs (int, optional): number of training epochs for autoencoder. Defaults to 100.
        batch_size (int, optional): Trainingsbatchsize. Defaults to 16.
        lr (float, optional): Learningrate. Defaults to 0.00005.
    """

    # paths = collect_image_dirs(data_dir_path, specific_dirs=specific_dirs)
    # paths = [p for p in paths if "results" not in p]
    # output_paths = collect_image_dirs(output_dir_path, specific_dirs=specific_dirs)    


    
   

    if SPECIFIC_DIRS:
        paths = collect_image_dirs(data_dir_path, SPECIFIC_DIRS)
        output_paths = collect_image_dirs(output_dir_path, SPECIFIC_DIRS)
    else:
        output_paths = output_dir_path
        paths = data_dir_path

    if type(paths) == str:
        paths = [paths]
    if type(output_paths) == str:
        output_paths = [output_paths]



    print(f"ALL PATHS: {paths}")
    print(f"ALL OUTPUT PATHS: {output_paths}")

    for i, data_path in enumerate(paths):
        print(f"#### Start Analysing {data_path}")
        print(all_IDs)
        print(np.arange(max_chunk_size,len(all_IDs),max_chunk_size))
        all_ids = np.split(all_IDs, np.arange(max_chunk_size,len(all_IDs),max_chunk_size))
        split_numbers = list(range(len(all_IDs)))
        
        
        
        if first_chunk==last_chunk:
            all_ids = all_ids
            split_numbers = list(split_numbers)
        else:
            all_ids = all_ids[first_chunk:last_chunk]
            split_numbers = split_numbers[first_chunk:last_chunk]
        

        for split_n, ids_to_keep in enumerate(all_ids):
            split_number = split_numbers[split_n]
            print(f"IDS TO KEEP: {ids_to_keep} from all IDs: {all_ids}")
            if not os.path.exists(output_paths[i]):
                os.mkdir(output_paths[i])
            doesnotyetexist = f"Embeddings_Meta_{split_number}.csv" not in os.listdir(output_paths[i])
            
            analysis = AnalyzeStudyAE(data_path, output_paths[i], split_number, ids_to_keep, TREATMENT_COLUMN, TARGET_COLUMN, csv_file=CSV_FILE)
                
            
            if doesnotyetexist or recreate:
                print("Computing embedding file.")
                
                data_loader, val_data_loader = analysis.create_data_loader(batch_size=batch_size)
                model, optimizer, loss_function, hist = analysis.init_model(lr=lr)
                model, hist_loss = analysis.train_model(
                    data_loader=data_loader, 
                    model=model, 
                    n_epochs=n_epochs, 
                    optimizer=optimizer, 
                    loss_function=loss_function, 
                    hist_loss=hist,
                    )

                del data_loader, val_data_loader

                all_org_images, all_reconst_images, all_meta_data = analysis.create_embeddings()

                del analysis, model, optimizer, hist_loss, all_org_images, all_reconst_images, all_meta_data
            else:
                print("Embedding file already exists. Skipping.")


if __name__=="__main__":
    main(
        data_dir_path=DATA_DIR_PATH,
        output_dir_path=OUTPUT_DIR_PATH,
        specific_dirs = SPECIFIC_DIRS,
       all_IDs = ALL_IDS, 
         max_chunk_size = MAX_CHUNK_SIZE, 
        first_chunk = FIRST_CHUNK, 
        last_chunk = LAST_CHUNK,
        recreate = RECREATE,
                    n_epochs=N_EPOCHS, 
    )
