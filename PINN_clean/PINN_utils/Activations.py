import torch
import torch.nn as nn



class Wavelet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)  
        self.w2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)      
    def forward(self, x):
        return self.w1*torch.sin(x) + self.w2*torch.cos(x)



class Sin_custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1    = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)  
        self.omega = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)      
    def forward(self, x):
        return self.w1*torch.sin(x*self.omega)




