
import numpy as np

import torch
import torch.nn as nn



class Gaus_Fourier_map(nn.Module):
    def __init__(self, N_vars = 2, mapping = 100, sigma = 5):
        super().__init__()

        B = torch.empty(N_vars*mapping).normal_(mean=0,std=sigma).reshape((N_vars,mapping))
        self.B = torch.nn.Parameter(B, requires_grad = False) #Freeze, making the B matrix trainable is a bad idea.
        
    def forward(self, x):        
        sin_ = torch.sin(x @ self.B)
        cos_ = torch.cos(x @ self.B)
        return torch.cat((sin_,cos_), dim =1)
    

