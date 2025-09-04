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

FINAL_IMAGE_SIZE = yaml_params["FINAL_IMAGE_SIZE"]
N_EPOCHS = yaml_params["N_EPOCHS"]
BATCH_SIZE = yaml_params["BATCH_SIZE"]
LEARNING_RATE = yaml_params["LEARNING_RATE"]

ALL_IDS = yaml_params["ALL_IDS"] 
MAX_CHUNK_SIZE = yaml_params["MAX_CHUNK_SIZE"] 
FIRST_CHUNK = yaml_params["FIRST_CHUNK"]
LAST_CHUNK = yaml_params["LAST_CHUNK"]
RECREATE = yaml_params["RECREATE"]
CSV_FILE = yaml_params["CSV_FILE"]

def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

def fast_scandir(dirname, specific_dirs=[]):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    subsubfolders = []
    for subdirname in list(subfolders):
        subsubfolders.extend([f.path for f in os.scandir(subdirname) if f.is_dir()])
    all_specific_subfolders = []
    for pair in specific_dirs:
        subsubsubfolders = subsubfolders.copy()
        for i in pair:
            subsubsubfolders = [s for s in subsubsubfolders if i in s]
        all_specific_subfolders.extend(subsubsubfolders)

    all_specific_subfolders = [s for s in all_specific_subfolders if ".ipynb_checkpoints" not in s]
    return all_specific_subfolders

def main(data_dir_path:str, output_dir_path:str, all_IDs: list, 
         max_chunk_size:int, first_chunk:int, last_chunk:int, specific_dirs:list, recreate=RECREATE, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE): 
    """Main funtion to create embeddings

    Args:
        data_dir_path (str): Path to data.
        output_dir_path (str): Path for the results.
        n_epochs (int, optional): number of trainingsepochs for autoencoder. Defaults to 100.
        batch_size (int, optional): Trainingsbatchsize. Defaults to 16.
        lr (float, optional): Learningrate. Defaults to 0.00005.
    """

    paths = fast_scandir(data_dir_path, specific_dirs=specific_dirs)
    paths = [p for p in paths if "results" not in p]
    output_paths = fast_scandir(output_dir_path, specific_dirs=specific_dirs)    

    for i, data_path in enumerate(paths):
        print(f"#### Start Analysing {data_path}")
        
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
            
            doesnotyetexist = f"Embeddings_Meta_{split_number}.csv" not in os.listdir(output_paths[i])
            
            analysis = AnalyzeStudyAE(data_path, output_paths[i], split_number, ids_to_keep, csv_file=CSV_FILE)
                
            
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
