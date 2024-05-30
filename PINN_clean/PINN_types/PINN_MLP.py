import numpy as np

import torch
import torch.nn as nn



from PINN_utils.Activations import Wavelet, Sin_custom


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


class PINN(torch.nn.Module):
    def __init__(self,
                input_dim,                  #   
                output_dim,                 #
                N_hidden  ,                 #
                width_layer,                #
                domains,                  
                PDE_funct,                  #
                Boundary_funct,             #
                Init_cond_funct,            # 
                mapping        = None,      #
                act            = "wavelet",
                device = device):
        super(PINN, self).__init__()

        '''
        One can add data driven loss by just going from 3 weights to 4 -->
        Write dynamic loss for the model for the data                  -->
        Modify the sampler so that you actually get the data points    -->
        Add the data driven loss to y vector in get_loss_dynamic       -->
        Weight balancing is quite straighforward to modify to accept data driven loss 
        also. It just follows the style of the IC and BC loss.

        '''
        self.weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
        
        self.PDE      = PDE_funct
        self.Initial  = Init_cond_funct
        self.Boundary = Boundary_funct

        self.domains = domains # Endpoint of domains to non dimenzionalize the data

        modules = []

        if mapping:
            map_output_dim = mapping(torch.rand(1,input_dim)).shape[-1] #Check the mapping dim 
            modules.append(mapping)
            input_dim = map_output_dim
        
        modules.append(nn.Linear(input_dim, width_layer))


        if act.lower() == "wavelet":
            modules.append(Wavelet())
        elif act.lower() == "sin":
            modules.append(Sin_custom())
        else:
            modules.append(nn.Tanh())


        for layer in range(N_hidden):
            modules.append(nn.Linear(width_layer,width_layer))
            if act.lower() == "wavelet":
                modules.append(Wavelet())
            elif act.lower() == "sin":
                modules.append(Sin_custom())
            else:
                modules.append(nn.Tanh())


        modules.append(nn.Linear(width_layer,output_dim))

        self.layers = nn.Sequential(*modules)
        

    # Keep the signature same as for NTK
    def forward(self, params, X): 
        return self.layers(X)
    
    

    def PDE_dynamic_loss(self, args):
        return torch.mean(self.PDE([],self,self.domains,  *args)**2)    #Return the PDE loss, empty params

    def PDE_dynamic_residual(self,params, args):
        return self.PDE(params,self, self.domains, *args)               #Return the PDE loss for NTK 
    
    def Boundary_dynamic(self, args):
        return self.Boundary(self, *args)                               #Return the Boundary loss 

    def Initial_dynamic(self, args):
        return self.Initial(self, *args)                                #Return Initial loss 

    def get_loss_dynamic(self, args1, args2, args3):        
        
        loss_pde = self.PDE_dynamic_loss(args1)
        loss_bc  = self.Boundary_dynamic(args2)
        loss_ic  = self.Initial_dynamic(args3)

        
        y = [loss_pde, loss_bc , loss_ic]
        ys = torch.stack(y,axis = 0)

        return ys


