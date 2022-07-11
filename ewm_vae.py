import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import create_dataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Model Hyperparameters

dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 10

"""
# 64 x 64
x_dim = 12288
hidden_dim = 6000
latent_dim = 3000
"""

# 32 x 32
x_dim = 3072
hidden_dim = 1600
latent_dim = 800


"""
# 28 x 28
x_dim = 2352    
hidden_dim = 1200
latent_dim = 600
"""

lr = 1e-3

epochs = 15

# Step 1. Load (or download) Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ewm_annot = '/home/campi120/ewm_vae/ewm_annotations.csv'
ewm_img_dir = '/home/campi120/ewm_vae/ewm_imgs'
ewm_dat = create_dataset.CustomImageDataset(ewm_annot,ewm_img_dir)


validation_split = .2
shuffle_dataset = True
random_seed= 42
# Creating data indices for training and validation splits:
dataset_size = 1000
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

train_loader = DataLoader(ewm_dat, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = DataLoader(ewm_dat, batch_size=batch_size,
                                                sampler=valid_sampler)


""""
# Step 2. Define our model: Variational AutoEncoder (VAE)
"""

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        fc1_ = self.FC_input(x.float())
        h_       = self.LeakyReLU(fc1_)
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        fc1 = self.FC_hidden(x)
        h     = self.LeakyReLU(fc1)
        h     = self.LeakyReLU(self.FC_hidden2(h))
        # print(h.size())
        x_hat = self.FC_output(h)
        #x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

#Step 3. Define Loss function (reprod. loss) and optimizer
from torch.optim import Adam

def loss_function(x, x_hat, mean, log_var):
    bce_loss = nn.BCEWithLogitsLoss()
    reproduction_loss = bce_loss(x_hat, x.float())
    KLD      = - 0.5 * torch.sum(1+ log_var - torch.pow(mean, 2) - torch.exp(log_var))

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)
#Step 4. Train Variational AutoEncoder (VAE)
print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        #print(x_hat.type(), mean.type(), log_var.type())
        
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")

with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

#save_image(generated_images.view(batch_size, 3, 28, 28), 'generated_sample.png')
save_image(generated_images.view(batch_size, 3, 32, 32), 'generated_sample.png')
#save_image(generated_images.view(batch_size, 3, 64, 64), 'generated_sample.png')



