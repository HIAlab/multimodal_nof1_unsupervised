import torch
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.xs = []
        self.res_data = []
        self.transform = transform

        prep_trans = Compose([
          Resize((300, 300)),
        ])

        for i, row in df.iterrows():
            #fn = row['ImageId']
            im = Image.open(f'{row["Path"]}/{row["ImageFile"]}'.replace("//","/").replace("./","/dhc/home/juliana.schneider/Simulate_Multimodal_Nof1/")).convert("RGB")# / fn[0] / fn)

            self.xs.append(prep_trans(im))

            relevant_columns = ['ImageId', 'Id','Intervention','Simulated_severity','ImageFile']
            self.res_data.append({c:row[c] for c in relevant_columns})
    

    def __getitem__(self, i):
        return self.transform(self.xs[i]), self.res_data[i]

    def __len__(self):
        return len(self.xs)
