import numpy as np

import torch
import torch.nn as nn



def PDE_convection(params,model, domains, x , t, beta):
    
    # du/dt + beta du/dx = 0
    x     = torch.autograd.Variable(x, requires_grad=True)
    t     = torch.autograd.Variable(t, requires_grad=True) 

    pde_X = torch.cat((x/domains[0],t/domains[1]), dim = -1)        

    if params:
        u = model(params, pde_X.reshape((2))) #Used with NTK
    else:
        u = model(params, pde_X)
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    residual = u_t + beta*u_x

    return residual




def PDE_wave(params, model, domains, x,t, c):
    
    x     = torch.autograd.Variable(x, requires_grad=True)
    t     = torch.autograd.Variable(t, requires_grad=True) 

    
    pde_X = torch.cat((x/domains[0],t/domains[1]), dim = -1)    

    if params:
        u = model(params, pde_X.reshape((2))) #Used with NTK
    else:
        u = model(params, pde_X)


    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    
    residual = u_tt - c*u_xx

    return residual

def PDE_convection_param_check(params,model, domains, x , t, beta, c, d ,y ):
    pd = c*d*y

    # du/dt + beta du/dx = 0
    x     = torch.autograd.Variable(x, requires_grad=True)
    t     = torch.autograd.Variable(t, requires_grad=True) 

    pde_X = torch.cat((x/domains[0],t/domains[1]), dim = -1)        

    if params:
        u = model(params, pde_X.reshape((2))) #Used with NTK
    else:
        u = model(params, pde_X)
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    residual = u_t + beta*u_x

    return residual