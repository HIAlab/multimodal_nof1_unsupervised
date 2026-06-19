import torch
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.config_utils import load_config

yaml_params = load_config()

RELEVANT_COLUMNS = yaml_params["RELEVANT_COLUMNS"]
SIMULATION_OR_PILOT = yaml_params["SIMULATION_OR_PILOT"]
if SIMULATION_OR_PILOT == "PILOT":
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_AVGSCORE"]
else:
    TARGET_COLUMN = yaml_params["TARGET_COLUMN_SIMSEV"]

RELEVANT_COLUMNS = RELEVANT_COLUMNS + [TARGET_COLUMN]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.xs = []
        self.res_data = []
        self.transform = transform

        prep_trans = Compose([
          Resize((300, 300)),
        ])

        for i, row in df.iterrows():
            im = Image.open(f'{row["Path"]}/{row["ImageFile"]}').convert("RGB")

            self.xs.append(prep_trans(im))

            self.res_data.append({c:row[c] for c in RELEVANT_COLUMNS})
    

    def __getitem__(self, i):
        return self.transform(self.xs[i]), self.res_data[i]

    def __len__(self):
        return len(self.xs)
