import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import load_config, parse_range_like, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create AE embeddings for pilot or simulation.")
    parser.add_argument("--config", default=None, help="pilot, sim, or path to a config yaml")
    return parser.parse_args()


ARGS = parse_args()
CONFIG = load_config(ARGS.config)

STDOUT_PATH = resolve_path(CONFIG["STDOUT_PATH"], CONFIG)
Path(STDOUT_PATH).parent.mkdir(parents=True, exist_ok=True)
sys.stdout = open(STDOUT_PATH, "w", buffering=1)

DATA_DIR_PATH = resolve_path(CONFIG["DATA_DIR_PATH"], CONFIG)
OUTPUT_DIR_PATH = resolve_path(CONFIG["OUTPUT_DIR_PATH"], CONFIG)
SPECIFIC_DIRS = CONFIG["SPECIFIC_DIRS"]
N_EPOCHS = CONFIG["N_EPOCHS"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]
LEARNING_RATE = CONFIG["LEARNING_RATE"]
MAX_CHUNK_SIZE = CONFIG["MAX_CHUNK_SIZE"]
FIRST_CHUNK = CONFIG["FIRST_CHUNK"]
LAST_CHUNK = CONFIG["LAST_CHUNK"]
RECREATE = CONFIG["RECREATE"]
TREATMENT_COLUMN = CONFIG["TREATMENT_COLUMN"]
SIMULATION_OR_PILOT = CONFIG["SIMULATION_OR_PILOT"]

ALL_IDS = parse_range_like(CONFIG["ALL_IDS"])
if isinstance(ALL_IDS, range):
    ALL_IDS = list(ALL_IDS)

if SIMULATION_OR_PILOT == "PILOT":
    TARGET_COLUMN = CONFIG["TARGET_COLUMN_AVGSCORE"]
    CSV_FILE = CONFIG["CSV_FILE_PILOT"]
else:
    TARGET_COLUMN = CONFIG["TARGET_COLUMN_PCA1"]
    CSV_FILE = CONFIG["CSV_FILE_SIMULATION"]

from src.EmbeddingsAE import AnalyzeStudyAE


def collect_image_dirs(data_dir_path: str, specific_dirs=None):
    specific_dirs = specific_dirs or []
    dataset_name = os.path.basename(data_dir_path.rstrip("/"))

    if dataset_name == "Acne_Nof1_trial":
        if not specific_dirs:
            return [data_dir_path]
        return [os.path.join(data_dir_path, str(d)) for d in specific_dirs]

    if dataset_name == "Simulation":
        if not specific_dirs:
            return [
                os.path.join(data_dir_path, d)
                for d in os.listdir(data_dir_path)
                if os.path.isdir(os.path.join(data_dir_path, d))
            ]
        return [os.path.join(data_dir_path, str(d)) for d in specific_dirs]

    raise ValueError(f"Unknown dataset type: {dataset_name}")


def main(data_dir_path: str, output_dir_path: str, all_ids: list, max_chunk_size: int, first_chunk: int, last_chunk: int, specific_dirs: list):
    if specific_dirs:
        paths = collect_image_dirs(data_dir_path, specific_dirs)
        output_paths = collect_image_dirs(output_dir_path, specific_dirs)
    else:
        paths = [data_dir_path]
        output_paths = [output_dir_path]

    for index, data_path in enumerate(paths):
        print(f"#### Start Analysing {data_path}")

        split_ids = np.split(all_ids, np.arange(max_chunk_size, len(all_ids), max_chunk_size))
        split_numbers = list(range(len(split_ids)))

        if first_chunk != last_chunk:
            split_ids = split_ids[first_chunk:last_chunk]
            split_numbers = split_numbers[first_chunk:last_chunk]

        for split_idx, ids_to_keep in enumerate(split_ids):
            split_number = split_numbers[split_idx]
            output_dir = Path(output_paths[index])
            output_dir.mkdir(parents=True, exist_ok=True)
            target_file = output_dir / f"Embeddings_Meta_{split_number}.csv"
            analysis = AnalyzeStudyAE(
                data_path,
                str(output_dir),
                split_number,
                ids_to_keep,
                TREATMENT_COLUMN,
                TARGET_COLUMN,
                csv_file=CSV_FILE,
            )

            if RECREATE or not target_file.exists():
                print("Computing embedding file.")
                data_loader, val_data_loader = analysis.create_data_loader(batch_size=BATCH_SIZE)
                model, optimizer, loss_function, hist = analysis.init_model(lr=LEARNING_RATE)
                model, hist_loss = analysis.train_model(
                    data_loader=data_loader,
                    model=model,
                    n_epochs=N_EPOCHS,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    hist_loss=hist,
                )
                del data_loader, val_data_loader
                all_org_images, all_reconst_images, all_meta_data = analysis.create_embeddings()
                del analysis, model, optimizer, hist_loss, all_org_images, all_reconst_images, all_meta_data
            else:
                print("Embedding file already exists. Skipping.")


if __name__ == "__main__":
    main(
        data_dir_path=DATA_DIR_PATH,
        output_dir_path=OUTPUT_DIR_PATH,
        all_ids=ALL_IDS,
        max_chunk_size=MAX_CHUNK_SIZE,
        first_chunk=FIRST_CHUNK,
        last_chunk=LAST_CHUNK,
        specific_dirs=SPECIFIC_DIRS,
    )
