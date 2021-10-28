import warnings

warnings.simplefilter(action='ignore') #, category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from Network import LinearAutoencoder
from pytorch_msssim import SSIM
import os
from tqdm import tqdm
from utils import set_seed_globally, dataSet, optimizer, test

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("USING GPU :)")
    device = 'cuda'
    pinMem = True
else:
    print("USING CPU :(")
    device = 'cpu'
    pinMem = False

set_seed_globally(seed_value=0, if_cuda=use_gpu)

data = "MNIST"
dataset_path = "./MNIST_Combined.npz"
Optimizer = "Adam"
learning_rate = 1e-3
batch_size = 512
channels = 1
alpha = 0.25
epochs = 200
log_dir = f"./log/{data}/BatchSize {batch_size} Optimizer {Optimizer} Learning_Rate {learning_rate} alpha {alpha}"
model_dir = f"./Model/{data}/BatchSize {batch_size} Optimizer {Optimizer} Learning_Rate {learning_rate} alpha {alpha}"
latent_dir = f"./Latent/{data}/BatchSize {batch_size} Optimizer {Optimizer} Learning_Rate {learning_rate} alpha {alpha}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

dataset = dataSet(dataset_path)



trainLoader = DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory = pinMem, prefetch_factor=batch_size//4)

model = LinearAutoencoder()

model.to(device=device)
criterion_mse = nn.MSELoss().to(device=device)
criterion_ssim = SSIM(channel=channels, data_range=1.0, size_average=True).to(device=device)


op = optimizer(model, lr=learning_rate, optimizer=Optimizer)
Optim = op.call()

for epoch in range(epochs):
    model.train()
    running_loss = 0
    loop = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False)
    for idx, (x,_,_) in loop:
        x = x.to(device)
        xt = x.reshape(x.size(0),-1)
        Optim.zero_grad()
        z , recon = model(xt)
        recon = recon.reshape(x.shape)

        mse = criterion_mse(recon, x)
        ssim = criterion_ssim(recon, x)
        loss = alpha * mse + (1 - alpha)*ssim

        running_loss += loss.item()
        loss.backward()
        Optim.step()
        loop.set_description(f"[{epoch}/{epochs}]")
        loop.set_postfix(Recon_loss=loss.item())
    
    if (epoch+1)%10 == 0:
        acc, nmi, ari = test(model, trainLoader, device=device, save_latents=latent_dir, epoch=epoch+1)
        print(f"Epoch:{epoch+1}, acc:{acc:.4f}, nmi:{nmi:.4f}, ari:{ari:.4f}")
        dict = {'weights': model.state_dict(),
                'optimizer': Optim.state_dict(),
                'epoch': epoch,
                'nmi': nmi,
                'acc': acc}
        torch.save(dict, f"{model_dir}/Epoch {epoch+1}.pth.tar")


