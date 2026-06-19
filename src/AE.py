import torch.nn as nn
import torch
# Model structure

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input image = 256x256
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, stride=4), # image 64x64
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 2, stride=2), # image 32x32
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ELU(),
            nn.Conv2d(64, 128, 2, stride=2), # image 16x16
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ELU(),
            nn.Conv2d(128, 256, 2, stride=2), # image 8x8
            nn.ELU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ELU(),
            nn.Conv2d(256, 256, 2, stride=2), # image 4x4
            nn.ELU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ELU(),
            nn.Conv2d(256, 512, 2, stride=2), # image 2x2
            nn.ELU(),
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ELU(),
            nn.Conv2d(512, 1024, 2, stride=2), # image 1x1
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ELU(),
            nn.Unflatten(-1, (64,4,4)),
            nn.ConvTranspose2d(64, 128, 2, stride=2), # 64x8x8
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2), # 64x16x16
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2), # 64x32x32
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), # 32x64x64
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), # 32x128x128
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), # 16x256x256
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded

