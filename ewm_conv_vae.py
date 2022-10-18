"""
Check and see what yolo pixel dim, 416 x 416
"""

from turtle import st
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import time

import torch.nn.functional as F

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import create_dataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import math

"""
Model Hyperparameters
"""
timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

n_pixels = 416
n_conv_layers = 3
batch_size = 1
lr = 1e-3
epochs = 10

# calculates feature dimensions after going through conv layer
def conv_layer_calc(n_layers, n_pixels, kernel, s = 2, p = 0):
    for n in range(n_layers):
        pix_next = ((math.floor(n_pixels+(2*p)-kernel))/s)+1
        n_pixels = pix_next
    return(int(pix_next))

post_conv_n_pixels = conv_layer_calc(2, n_pixels, kernel=3,s=2, p =1)
post_conv_n_pixels
# these will be dimensions of FC layers after the 3 conv layers
feature_dim = 32*post_conv_n_pixels*post_conv_n_pixels
z_dim = 768

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
 Step 1. Load (or download) Dataset
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ewm_annot = './ewm_annotations.csv'
ewm_img_dir = './ewm_imgs'
ewm_dat = create_dataset.CustomImageDataset(ewm_annot,ewm_img_dir, img_dim=(n_pixels,n_pixels))


validation_split = .1
shuffle_dataset = True
random_seed= 42
# Creating data indices for training and validation splits:
dataset_size = 900
#dataset_size = len(ewm_dat)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(ewm_dat, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(ewm_dat, batch_size=batch_size, sampler=valid_sampler)

print('step 1 done')
print(DEVICE)

"""
Step 2. Define our model: Convolutional Variational Autoencoder
Source from https://github.com/ttchengab/VAE/blob/main/VAE.py
"""
class VAE(nn.Module):
    # imgChannel is 1 for grayscale, 3 for RGB
    # featureDim is for FC layers, calculated in step 1
    # z_dim should be divisible by 16 and maybe ~2% size of feature dim
    def __init__(self, imgChannels=3, featureDim=32*31*31, zDim=768):
        super(VAE, self).__init__()
    
        self.featureDim = featureDim
        self.zDim = zDim
        # Initializing the 3 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 8, 3, stride=2, padding=1) # will decrease img dim to 8*127*127
        self.encConv2 = nn.Conv2d(8, 16, 3, stride=2, padding = 1) # will decrease img dim to 16*63*63
        self.encConv3 = nn.Conv2d(16, 32, 3, stride=2, padding = 1) # will decrease img dim to 32*31*31
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 3,stride=2, padding = 1, output_padding = 1)
        self.decConv2 = nn.ConvTranspose2d(16, 8, 3,stride=2, padding = 1, output_padding =1)
        self.decConv3 = nn.ConvTranspose2d(8, imgChannels, 3,stride=2, padding = 1, output_padding=1)


    def encoder(self, x):
        # Input is fed into 3 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mean) and variance (log_var)
        # mean and log_var are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = x.view(x.size(0), -1)
        mean = self.encFC1(x)
        log_var = self.encFC2(x)
        return mean, log_var

    def reparameterize(self, mean, log_var):

        #Reparameterization takes in the input mean and log_var and sample the mean + std * eps
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into 3 transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, post_conv_n_pixels, post_conv_n_pixels)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mean, and log_var are returned for loss computation
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

model = VAE(3, feature_dim, z_dim).to(DEVICE)

"""
Step 3. Define Loss function (reprod. loss) and optimizer
"""
"""
def loss_function(x, x_hat, mean, log_var):
    kl_divergence = -0.5 * torch.sum(1 + log_var -torch.pow(mean, 2) - torch.exp(log_var))
    #print(x_hat.size())
    #print(x.size())
    loss = F.binary_cross_entropy(x_hat, x, reduction='sum') 
    #print(loss.type)
    return(loss+ kl_divergence)
"""

def loss_function(x, x_hat, mean, log_var):
    bce_loss = nn.BCEWithLogitsLoss()
    reproduction_loss = bce_loss(x_hat, x.float())
    KLD      = - 0.5 * torch.sum(1+ log_var - torch.pow(mean, 2) - torch.exp(log_var))
    return reproduction_loss + KLD

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

"""
Step 4. Train Convolutional VAE
"""
print("Start training VAE...")
model.train()
loss_arr = [None]*epochs
vloss_arr = [None]*epochs
for epoch in range(epochs):
    print('\tEpoch', epoch + 1, 'starting')
    overall_loss = 0
    for idx, data in enumerate(train_loader):
        x, _ = data
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)

        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = overall_loss / ((idx+1)*batch_size)
    loss_arr[epoch] = avg_loss
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", avg_loss)

    # apply current weights to validation set
    model.eval()
    overall_vloss = 0
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
                x, _ = data
                x = x.to(DEVICE)
                
                x_hat, _, _ = model(x)

                # calculate validation loss
                vloss = loss_function(x, x_hat, mean, log_var)
                overall_vloss += vloss.item()

                # we don't need to save all 100 images every iteration, lets only save 10
                if((idx+1)%10 == 0):
                    img_pathname = './val_img_output/ewm_val_imgs_epoch_' + str(epoch) +'_'+str(idx)+'.jpg'
                    save_image(x.view(batch_size, 3, n_pixels, n_pixels), img_pathname)
        # calculate average validation loss, store on avg_vloss array
        avg_vloss = overall_vloss/((idx+1)*batch_size)
        vloss_arr[epoch] = avg_vloss
        print("\tEpoch", epoch + 1, "\Averge Validation Loss: ", avg_vloss)

    if ((epoch+1) % 5 == 0):
        print('Saving model')
        # save model.state,save as ewm_vae_weight_'epoch#'
        model_pathname = './aug2_weight_outputs/ewm_model_weights_epoch_' + str(epoch) +'.pth'
        torch.save(model.state_dict(), model_pathname)
 
              
print("Finish!!")
# save model loss and validation model loss
loss_df = pd.DataFrame(loss_arr)
loss_df.to_csv('./loss_arr_'+timestr+'.csv')
vloss_df = pd.DataFrame(vloss_arr)
vloss_df.to_csv('./vloss_arr_'+timestr+'.csv')

