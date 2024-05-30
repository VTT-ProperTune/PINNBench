import numpy as np
import torch
import torch.nn as nn




class Classic:
    def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer

            '''
            These are here so that the LBFGS optimizer can work with the function train_cycle 
            '''
            self.points_pde = None
            self.points_bc  = None
            self.points_ic  = None
            self.losses     = None

    def train_cycle(self):    

        self.optimizer.zero_grad()
        loss = self.model.get_loss_dynamic(self.points_pde, self.points_bc, self.points_ic)
        loss_w = self.model.weights @ loss
        
        loss_w.backward()                
        self.losses = loss

        #self.optimizer.step()

        return loss_w





class Causality_weighting:
    def __init__(self, model, optimizer,device, epsilon = 0.5):
        self.model     = model
        self.optimizer = optimizer
        self.epsilon   = epsilon
        self.device    = device
        
        self.A = torch.tril(torch.ones((2500,2500))).to(self.device)   #System matrix 

        '''
        These are here so that the LBFGS optimizer can work with the function train_cycle 
        '''
        self.points_pde = None
        self.points_bc  = None
        self.points_ic  = None
        self.losses     = None

    


    def train_cycle(self):
        self.optimizer.zero_grad()
        pde_residual = self.model.PDE_dynamic_residual([], self.points_pde)**2

        w = torch.exp((-self.epsilon*self.A @ pde_residual))   #causal weights

        loss_pde = torch.mean(w*pde_residual)
        
        loss_bc  = self.model.Boundary_dynamic(self.points_bc)
        loss_ic  = self.model.Initial_dynamic(self.points_ic)


        y = [loss_pde, loss_bc , loss_ic]
        ys = torch.stack(y,axis = 0)

        loss = ys

        loss_w = self.model.weights @ loss

        loss_w.backward()                

        self.losses = ys
        
        #self.optimizer.step()

        return loss_w


    


                
