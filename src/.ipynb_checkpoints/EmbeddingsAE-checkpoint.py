from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from src.AE import AutoEncoder
from src.ImageDataset import ImageDataset
from src.AE_utils import default_trans, non_trans


class AnalyzeStudyAE():
    def __init__(self, data_path, output_dir, split_number, ids_to_keep,csv_file="study.csv"):
        self.output_dir = output_dir
        self.data_path = data_path
        self.split_number = split_number
        self.study_df = self._read_study_params(csv_file,ids_to_keep)

    def _read_study_params(self, study_file_path,ids_to_keep):
        sim_df = pd.read_csv(f"{self.data_path}/{study_file_path}")
        df = []
        
        sim_ids = sim_df["Id"].unique()
        ids_to_keep= [sim_ids[i] for i in ids_to_keep]
        for i in ids_to_keep: 
            id_df = sim_df[sim_df["Id"] == i].copy()
            phases = id_df["Intervention"] 
            intervention = id_df["Intervention"] == id_df["Intervention"].unique()[0]

            id_df["phase"] = phases
            id_df["Intervention"] = intervention
            df.append(id_df)
            
        df = pd.concat(df).sort_values(["Id", "ImageId"])
        print(f"{df.Id.nunique()} Unique IDs: {df.Id.unique()}")
        return df

    def create_data_loader(self, batch_size):
        df=self.study_df
        train_dataset = ImageDataset(df.copy(), default_trans)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = ImageDataset(df.copy(), non_trans)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return data_loader, val_data_loader


    def init_model(self, lr=0.0001):
        model = AutoEncoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        hist=[]
        return model, optimizer, loss_function, hist


    def train_model(self, data_loader, model, n_epochs, optimizer, loss_function,hist_loss=[], cross_validation_factor=0.75):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        model.to(device)

        start_epoch = int(len(hist_loss) / len(data_loader))
        def _step(data, model, bool_train):         
            inputs = data[0].to(device)
            # Forward
            codes, decoded = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss = loss_function(decoded, inputs)
            loss.backward()
            if bool_train:
                optimizer.step()
            return model, loss.item()

        for n_ep, epoch in enumerate(range(n_epochs)):
            print(f"Epoch number: {n_ep}")
        #    loop = tqdm(data_loader)
            for i, data in enumerate(data_loader):#(loop):
                
                steps_per_epoch = len(data_loader)
                bool_train = i <= int(cross_validation_factor *  steps_per_epoch)
                model, loss = _step(data, model, bool_train)
                hist_loss.append(
                    {
                        "Epoch": epoch + start_epoch +1, 
                        "tp": datetime.now(),
                        "loss": loss, 
                        "train_valid": "train" if bool_train else "valid"})
                train_losses = [e["loss"] for e in hist_loss[-(i+1):] if e["train_valid"]=="train"]
                valid_losses = [e["loss"] for e in hist_loss[-(i+1):] if e["train_valid"]=="valid"]

        torch.save(model, f'{self.output_dir}/.temp_autoencoder_split_{self.split_number}.pth')
        pd.DataFrame(hist_loss).to_csv(f'{self.output_dir}/.temp_hist_split_{self.split_number}.csv')
        self.model=model
        self.hist_loss=hist_loss
        return model, hist_loss


    def create_embeddings(self):
        # Create empty results
        org_images = []
        res_images = []
        embeddings = []
        meta_data = []
        # Model to cpu (To have everything on the same device)
        # TODO: Allow GPU
        self.model.to("cpu")
        
        # Iterate over all images in study_df
        for i, row in self.study_df.iterrows():#tqdm(self.study_df.iterrows()):
            # Load Image
            img = Image.open(f'{row["Path"]}/{row["ImageFile"]}'.replace("//","/").replace("./","~/Simulate_Multimodal_Nof1/")).convert("RGB").resize(FINAL_IMAGE_SIZE)
            img = non_trans(img)
            img = img.unsqueeze(0)
            
            # Predict
            _emb, _res = self.model(img)
             
            # Append results
            relevant_columns = ['ImageId', 'Id','Intervention','Simulated_severity','ImageFile']
            meta_data.append({c:row[c] for c in relevant_columns})
            org_images.append(img.detach().numpy())
            res_images.append(_res.detach().numpy())
            embeddings.append(_emb.detach().numpy())

        # Concatenate Results
        all_org_images = np.concatenate(org_images, axis=0)
        all_reconst_images = np.concatenate(res_images, axis=0)
        all_embeddings = np.concatenate(embeddings, axis=0)
        all_meta_data = pd.DataFrame(meta_data)
        embeddings = pd.DataFrame(all_embeddings, columns=[f"Embedding_{i+1}"for i in range(all_embeddings.shape[1])])
        all_meta_data = all_meta_data.join(embeddings)
        # Create reconstruction error
        reconstruction_error = ((all_org_images - all_reconst_images)**2).mean(axis=(1,2,3))
        all_meta_data["Reconstruction Error"] = reconstruction_error
        # Save embeddings file
        all_meta_data.to_csv(f"{self.output_dir}/Embeddings_Meta_{self.split_number}.csv".replace("[","").replace("]",""))
        return (all_org_images, all_reconst_images, all_meta_data)
