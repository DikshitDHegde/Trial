import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from torchsummary import summary

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,2000),
            nn.ReLU(),
            nn.Linear(2000,10)
        )

        self.dec = nn.Sequential(
            nn.Linear(10,2000),
            nn.ReLU(),
            nn.Linear(2000,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z =self.enc(x)
        recon = self.dec(z)
        return z, recon


def testLinearAutoencoder():
    input_ = torch.rand((1,1,28,28),dtype=torch.float)
    model = LinearAutoencoder()
    summary(model, (1,784),device="cpu")
    output = model(input_.reshape(input_.size(0),-1))
    print(output)

if __name__ == "__main__":
    testLinearAutoencoder()