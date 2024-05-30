import numpy as np

import torch
import torch.nn as nn



class sample_domain_dynamic:
    '''
    Resample for each epoch
    '''
    def __init__(self, device, x_domain, t_domain):
        np.random.seed(123)
        self.ts = torch.linspace(0,t_domain,5000).to(device)
        self.xs = torch.linspace(0,x_domain,5000).to(device)
        
        
        self.x_2pi          = torch.ones(250).to(device).reshape((-1,1))*torch.pi*2            
        self.zeros_boundary = torch.zeros(250).to(device).reshape((-1,1))


    def get_sample(self):
        x_pde = self.xs[np.random.choice(range(5000), size = 2500, replace= False)].reshape((-1,1))
        t_pde = self.ts[np.random.choice(range(5000), size = 2500, replace= False)].reshape((-1,1))

        t_bound = self.ts[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))
        x_bound = self.xs[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))
        
        #PDE_points, BC_points, IC_points
        return [x_pde, t_pde], [self.zeros_boundary, t_bound , self.x_2pi, t_bound], [x_bound, self.zeros_boundary]



class sample_domain_static:
    '''
    This is used when using causal training scheme as the time axis has to be ordered.
    '''
    def __init__(self, device, x_domain, t_domain):
        np.random.seed(123)
        self.ts = torch.linspace(0,t_domain ,5000).to(device)
        self.xs = torch.linspace(0,x_domain,5000).to(device)
        
        
        self.x_2pi          = torch.ones(250).to(device).reshape((-1,1))*torch.pi*2            
        self.zeros_boundary = torch.zeros(250).to(device).reshape((-1,1))


        self.x_pde = self.xs[np.random.choice(range(5000), size = 2500, replace= False)].reshape((-1,1))
        self.t_pde = self.ts[np.random.choice(range(5000), size = 2500, replace= False)].reshape((-1,1))


        self.t_bound = self.ts[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))
        self.x_bound = self.xs[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))

        
        ind =  self.t_pde[:,-1].argsort(dim=0) # sort t dim
        #sorted
        self.t_pde = self.t_pde[ind]


        
    def get_sample(self):
        #x_pde = self.xs[np.random.choice(range(5000), size = 2500, replace= False)].reshape((-1,1))        
        #t_bound = self.ts[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))
        #x_bound = self.xs[np.random.choice(range(5000), size = 250, replace= False)].reshape((-1,1))

        #PDE_points, BC_points, IC_points

        return [self.x_pde, self.t_pde], [self.zeros_boundary, self.t_bound , self.x_2pi, self.t_bound], [self.x_bound, self.zeros_boundary]



#https://github.com/AdityaLab/pinnsformer/blob/main/util.py
#https://arxiv.org/abs/2307.11833

class Transformer_sampler_static:
    def __init__(self, device, x_domain, t_domain):
        np.random.seed(123)

        self.device = device

        x = np.linspace(0, x_domain, 50)
        t = np.linspace(0,t_domain, 50)

        self.t_domain = t_domain
        self.x_domain = x_domain

        x_mesh, t_mesh = np.meshgrid(x,t)
        data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)

        initial = data[0,:,:] 

        boundary1 = data[:,-1,:]
        boundary2 = data[:,0,:]
        res = data.reshape(-1,2)


        res       = self.make_time_sequence(res, num_step=5, step=1e-4)
        initial   = self.make_time_sequence(initial, num_step=5, step=1e-4)        
        boundary1 = self.make_time_sequence(boundary1, num_step=5, step=1e-4)
        boundary2 = self.make_time_sequence(boundary2, num_step=5, step=1e-4)


        self.x_pde    , self.t_pde       = res[:,:,0:1], res[:,:,1:2]
        self.x_initial ,self.t_initial    = initial[:,:,0:1], initial[:,:,1:2]        
        self.x_bound1 , self.t_bound1    = boundary1[:,:,0:1], boundary1[:,:,1:2]
        self.x_bound2 ,self.t_bound2     = boundary2[:,:,0:1], boundary2[:,:,1:2]
        
        #Transfer to GPU tensor
        self.x_pde    = torch.from_numpy(self.x_pde).type(torch.float).to(device)
        self.t_pde    = torch.from_numpy(self.t_pde).type(torch.float).to(device)

        self.x_initial = torch.from_numpy(self.x_initial).type(torch.float).to(device)
        self.t_initial = torch.from_numpy(self.t_initial).type(torch.float).to(device)

        self.x_bound1 = torch.from_numpy(self.x_bound1).type(torch.float).to(device)
        self.t_bound1 = torch.from_numpy(self.t_bound1).type(torch.float).to(device)
        self.x_bound2 = torch.from_numpy(self.x_bound2).type(torch.float).to(device)
        self.t_bound2 = torch.from_numpy(self.t_bound2).type(torch.float).to(device)


    def make_time_sequence(self,src, num_step=5, step=1e-4):
        dim = num_step
        src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
        for i in range(num_step):
            src[:,i,-1] += step*i
        return src


    def get_sample(self):
        return [self.x_pde, self.t_pde], [self.x_bound1, self.t_bound1 , self.x_bound2, self.t_bound2], [self.x_initial, self.t_initial]
    

    def get_prediction_grid(self):

        x = np.linspace(0, self.x_domain, 250)
        t = np.linspace(0,self.t_domain, 250)

        x_mesh, t_mesh = np.meshgrid(x,t)
        data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
        res = data.reshape(-1,2)

        res       = self.make_time_sequence(res, num_step=5, step=1e-4)

        x_pde    , t_pde       = res[:,:,0:1], res[:,:,1:2]

        #Transfer to GPU tensor
        x_pde    = torch.from_numpy(x_pde).type(torch.float).to(self.device)
        t_pde    = torch.from_numpy(t_pde).type(torch.float).to(self.device)
        return x_pde, t_pde

