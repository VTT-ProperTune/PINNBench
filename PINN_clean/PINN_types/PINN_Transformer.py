import numpy as np

import torch
import torch.nn as nn

import copy

from PINN_utils.Activations import Wavelet



#https://github.com/AdityaLab/pinnsformer/blob/main/model/pinnsformer.py
#https://arxiv.org/abs/2307.11833

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            Wavelet(),
            nn.Linear(d_ff, d_ff),
            Wavelet(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)
    

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = Wavelet()
        self.act2 = Wavelet()
        
    def forward(self, x):
        x2 = self.act1(x)

        x = x + self.attn(x2,x2,x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = Wavelet()
        self.act2 = Wavelet()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = Wavelet()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = Wavelet()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)



class PINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, domains, PDE_funct, Boundary_funct, Init_cond_funct):
        super(PINNsformer, self).__init__()

        self.PDE      = PDE_funct
        self.Boundary = Boundary_funct
        self.Initial  = Init_cond_funct

        self.domains = domains


        self.weights = torch.tensor([1.0,1.0,1.0]).to(device)

        self.linear_emb = nn.Linear(2, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            Wavelet(),
            nn.Linear(d_hidden, d_hidden),
            Wavelet(),
            nn.Linear(d_hidden, d_out)
        ])


        self.layers = self.linear_out # for gradnorm


    def forward(self, params, src):
        #src = torch.cat((x,t), dim=-1)
        src = self.linear_emb(src)

        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)

        return output
    

    def PDE_dynamic_loss(self, args):
        return torch.mean(self.PDE([],self, self.domains, *args)**2)    #Return the PDE loss, empty params

    def PDE_dynamic_residual(self,params, args):
        return self.PDE(params,self, self.domains, *args)               #Return the PDE loss for NTK 
    
    def Boundary_dynamic(self, args):
        return self.Boundary(self,*args)                  #Return the Boundary loss 

    def Initial_dynamic(self, args):
        return self.Initial(self,*args)  #Return Initial loss 


    def get_loss_dynamic(self, args1, args2, args3):        


        loss_pde = self.PDE_dynamic_loss(args1)
        loss_bc  = self.Boundary_dynamic(args2)
        loss_ic  = self.Initial_dynamic(args3)


        y = [loss_pde, loss_bc , loss_ic]
        ys = torch.stack(y,axis = 0)

        return ys


