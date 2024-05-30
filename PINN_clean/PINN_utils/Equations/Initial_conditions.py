import numpy as np

import torch
import torch.nn as nn



def IC_funct(model, x1, t1, beta):
    X_IC  = torch.cat((x1/model.domains[0], t1/model.domains[1]), dim = -1)
    loss_ic  = torch.mean((model.forward(None, X_IC)[:,0] - torch.sin(x1[:,0]))**2)    #Scale the x_domain, t is zero anyways 
    return loss_ic



def IC_funct_wave(model, x1,t1, c):

    #analytical = np.sin(np.pi*X)*np.cos(2*np.pi*T)+1/2*np.sin(c*np.pi*X)*np.cos(2*c*np.pi*T)

    x     = torch.autograd.Variable(x1, requires_grad=True)
    t     = torch.autograd.Variable(t1, requires_grad=True) 

    X_IC  = torch.cat((x/model.domains[0], t/model.domains[1]), dim = -1)
    
    u = model.forward(None, X_IC)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    
    loss_ic1  = torch.mean((u[:,0]   - (torch.sin(torch.pi*x1[:,0])+1/2*torch.sin(torch.pi*x1[:,0]*c)))**2)       #u(ic)   = f(x)
    loss_ic2  = torch.mean(u_t**2)                                                                                #u_t(ic) = g(x)=0

    return loss_ic1+loss_ic2