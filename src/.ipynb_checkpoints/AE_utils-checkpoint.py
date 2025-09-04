from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, ColorJitter, RandomCrop, RandomEqualize, RandomVerticalFlip


default_trans = Compose([
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    RandomResizedCrop(size=FINAL_IMAGE_SIZE, scale=(0.8,1.0), ratio=(0.9,1.1)),
    Resize(FINAL_IMAGE_SIZE),
    ToTensor(),
])

non_trans = Compose([
    Resize(FINAL_IMAGE_SIZE),
    ToTensor(),
])