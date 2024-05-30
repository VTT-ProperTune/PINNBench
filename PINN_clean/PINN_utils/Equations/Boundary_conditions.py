import numpy as np

import torch
import torch.nn as nn


def BC_equal(model, x1, t1, x2,t2):

    X_bound0  = torch.cat((x1/model.domains[0], t1/model.domains[1]), dim = -1)
    X_bound1  = torch.cat((x2/model.domains[0], t2/model.domains[1]), dim = -1)
    
    bound_0 = model.forward(None,X_bound0)       # u(0,t) 
    bound_1 = model.forward(None,X_bound1)       # u(2pi, t), u(0,t) = u(2pi, t)

    loss_bc  = torch.mean((bound_0-bound_1)**2)
    return loss_bc

def BC_zeros(model, x1, t1, x2,t2):
    X_bound0  = torch.cat((x1/model.domains[0], t1/model.domains[1]), dim = -1)
    X_bound1  = torch.cat((x2/model.domains[0], t2/model.domains[1]), dim = -1)

    bound_0 = model.forward(None,X_bound0)       # u(0,t) 
    bound_1 = model.forward(None,X_bound1)       # u(2pi, t), u(0,t) = u(2pi, t)
    
    loss_bc  = torch.mean(bound_0**2+bound_1**2)
    return loss_bc
