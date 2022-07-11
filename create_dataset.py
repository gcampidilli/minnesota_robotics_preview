import os
from pickle import TRUE
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
import cv2
import numpy as np
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_dim = (32,32), target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, names=['img_fname','class'])
        self.img_dir = img_dir
        self.img_dim = img_dim        
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.img_dim)
        # convert BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # define a transform to convert image to 
        # torch tensor
        trfm = transforms.Compose([transforms.ToTensor()])

        # convert image to torch tensor
        image_tensor = trfm(image)
      
        if self.target_transform:
            label = self.target_transform(label)
        return image_tensor, label
    


                                                






