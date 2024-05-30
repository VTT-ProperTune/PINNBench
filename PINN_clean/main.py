import PINN_types.PINN_MLP
import PINN_types.PINN_Transformer
import PINN_utils.Domain_sampler
import PINN_utils.Equations
import PINN_utils.Equations.Boundary_conditions
import PINN_utils.Equations.Initial_conditions
import PINN_utils.Equations.PDE_equations
import PINN_utils.Mappings
import PINN_utils.Training_schemes
import PINN_utils.Weight_balancing
import numpy as np

import torch
import torch.nn as nn


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



import PINN_utils
import PINN_types

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


class Main:
    def __init__(self,        
                 model,
                 epoch = 80000,
                 alpha_lr = 0.90, 
                 lr_init = 1e-3,
                 min_lr  = 1e-9,
                 lr_freq = 5000,      
                 weight_balance_freq = 2500,
                 weight_balance     = "NTK2",
                 optimizer          = "adam",
                 training_scheme    = "causal",
                 sampler_type       = "static",
                 sampler_domain     = [1,1], # t, x domain
                 log  =True,
                 PDE_coefs = [70],
                 alpha = 0.9,
                 epsilon = 0.1,
                 gradnorm_coefs = [1.5, 100, 1e-3]):
        super(Main, self).__init__()

        self.weight_balance = weight_balance

        self.model = model

        if optimizer == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), line_search_fn ='strong_wolfe')
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_init)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)

        self.N_epochs  = epoch

        self.PDE_coefs = PDE_coefs #Extra coefs such as convection coef 


        if sampler_type.lower() == "transformer":
            self.transformer_flag = True
        else:
            self.transformer_flag = False

        self.optimizer_name = optimizer


        if sampler_type.lower() == "static":
            self.sampler = PINN_utils.Domain_sampler.sample_domain_static(device, *sampler_domain)

        elif sampler_type.lower() =="transformer":
            self.sampler = PINN_utils.Domain_sampler.Transformer_sampler_static(device, *sampler_domain)

        else:
            self.sampler = PINN_utils.Domain_sampler.sample_domain_dynamic(device, *sampler_domain)


        if weight_balance == "NTK":                        
            self.NTK = PINN_utils.Weight_balancing.NTK(model, model.PDE, self.PDE_coefs, device, self.sampler, alpha = alpha)            

        elif weight_balance == "NTK2":
            self.NTK = PINN_utils.Weight_balancing.NTK2(model, device, self.sampler, alpha = alpha)
            
        elif weight_balance and weight_balance.lower()  == "grad":            
            self.Grad = PINN_utils.Weight_balancing.GradNorm(model, alpha=gradnorm_coefs[0], T = gradnorm_coefs[1],lr = gradnorm_coefs[2])            


        if training_scheme.lower() == "classic" or sampler_type.lower() =="transformer":            
            self.training_scheme = PINN_utils.Training_schemes.Classic(self.model, self.optimizer)
        else:
            self.training_scheme = PINN_utils.Training_schemes.Causality_weighting(self.model, self.optimizer, device=device, epsilon = epsilon)


        self.lr_freq = lr_freq

        self.weight_balance_freq = weight_balance_freq

        self.log = log

        self.weight_history = {"PDE":[], "BC":[], "IC":[]}
        self.loss_history   = {"PDE":[], "BC":[], "IC":[]}
        self.lr_history     = [lr_init]


        INITIAL_LEARNING_RATE = lr_init

        your_min_lr = min_lr

        lambda1 = lambda epoch: max(alpha_lr** epoch, your_min_lr / INITIAL_LEARNING_RATE)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.min_lr = min_lr


    def train_PINN(self):
        for epoch in range(self.N_epochs):
            points_pde, points_bc, points_ic = self.sampler.get_sample()   #Sampler.get_sample()
            points_pde.extend(self.PDE_coefs)                              #args_pde #x_pde, t_pde, coefs
            #points_bc.extend(self.PDE_coefs)                              #add coefs
            points_ic.extend(self.PDE_coefs)

            '''
            This is done in this way so that the train cycle matches closure() of LBFGS at the same time
            '''
            self.training_scheme.points_pde = points_pde
            self.training_scheme.points_bc  = points_bc
            self.training_scheme.points_ic  = points_ic

            if self.optimizer_name == "LBFGS":
                self.optimizer.step(self.training_scheme.train_cycle)
            else:
                loss = self.training_scheme.train_cycle()
                self.optimizer.step()

            if epoch%self.lr_freq ==0:
                if not self.optimizer_name == "LBFGS":
                    self.scheduler.step()                            
                if self.log:
                    self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                    
            if epoch%self.weight_balance_freq == 0:

                if self.weight_balance == "NTK" or self.weight_balance == "NTK2":                    
                    self.NTK.balance()

                elif self.weight_balance and self.weight_balance.lower() =="grad":
                    loss = self.model.get_loss_dynamic(points_pde, points_bc, points_ic)
                    self.Grad.normalize_grad(loss)
                                        

                if self.log:                            
                    loss = self.training_scheme.losses

                    self.weight_history["PDE"].append(self.model.weights[0].detach().cpu())
                    self.weight_history["BC"].append(self.model.weights[1].detach().cpu())
                    self.weight_history["IC"].append(self.model.weights[2].detach().cpu())


                    self.loss_history["PDE"].append(loss[0].detach().cpu())
                    self.loss_history["BC"].append(loss[1].detach().cpu())
                    self.loss_history["IC"].append(loss[2].detach().cpu())

    def prediction(self):
        if self.transformer_flag:
            x_pde, t_pde = self.sampler.get_prediction_grid() #1000x1000 grid use CPU
            X = torch.cat((x_pde/self.model.domains[0], t_pde/self.model.domains[1]), dim = -1)
            with torch.no_grad():
                pred = self.model.forward(None, X)[:,0:1]
                #pred = pred.cpu().detach().numpy()                
            return pred.reshape(250,250)
        else:
            predicted = torch.rand((1000,1000))
            self.model_cpu = self.model.cpu()
            self.model_cpu.eval()

            with torch.no_grad():    
                for i, ti in enumerate(np.linspace(0,1,1000)):
                    X = torch.cat((torch.linspace(0,1.0,1000).reshape((-1,1)), ti*torch.ones(1000).reshape((-1,1))), dim =1)
                    predicted[i,:]  = self.model_cpu(None,X)[:,0]
            return predicted

    
    def plot_history(self, name):
        
        fig, axes = plt.subplots(3, 3, figsize=(16,10))

        axes[0,0].plot(self.loss_history["PDE"])
        axes[0,1].plot(self.loss_history["BC"])
        axes[0,2].plot(self.loss_history["IC"])


        axes[0,0].set_yscale('log')
        axes[0,1].set_yscale('log')
        axes[0,2].set_yscale('log')

        
        axes[1,0].set_yscale('log')
        axes[1,1].set_yscale('log')
        axes[1,2].set_yscale('log')


        axes[1,0].plot(np.array(self.weight_history["PDE"])*np.array(self.loss_history["PDE"]))
        axes[1,1].plot(np.array(self.weight_history["BC"])*np.array(self.loss_history["BC"]))
        axes[1,2].plot(np.array(self.weight_history["IC"])*np.array(self.loss_history["IC"]))


        axes[2,0].plot(self.weight_history["PDE"])
        axes[2,1].plot(self.weight_history["BC"])
        axes[2,2].plot(self.weight_history["IC"])

        axes[0,0].set_title("Loss PDE")
        axes[0,1].set_title("Loss BC")
        axes[0,2].set_title("Loss IC")

        axes[1,0].set_title("Weighted PDE")
        axes[1,1].set_title("Weighted BC")
        axes[1,2].set_title("Weighted IC")

        axes[2,0].set_title("lambda PDE")
        axes[2,1].set_title("lambda BC")
        axes[2,2].set_title("lambda IC")


        plt.savefig(name)


    def plot_log(self, run_name):
        run_name_np = "TEST_soft/" +"TEST_np/" +  run_name
        run_name = "TEST_soft/" +  run_name
        self.plot_history(run_name + "_history.png") #Plot loss/weight history

        predicted = self.prediction()                 #Get prediction on domain
        fig = plt.figure(figsize=(9, 5))            
        ax = fig.add_subplot(111)

        img = plt.imshow(np.transpose(predicted.detach().to("cpu")), interpolation = "nearest", cmap = "magma", extent=[0, 1, 0, 2*np.pi],
        origin='lower', aspect='auto', vmin=-1, vmax=1)

        np.save(run_name_np + ".npy", predicted.detach().to("cpu").numpy())
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=15)

        ax.set_xlabel('t', fontweight='bold', size=15)
        ax.set_ylabel('x', fontweight='bold', size=15)

        ax.tick_params(labelsize=15)

        plt.savefig(run_name + "_prediction.png")

        fig = plt.figure(figsize=(16,10))

        iter = 0
        lr_history = []
        
        for k in range(self.N_epochs):
            lr_history.append(self.lr_history[iter])
            if k%self.lr_freq == 0:
                iter+=1
        
        plt.plot(lr_history, label="lr", color="red")
        plt.yscale("log")
        plt.axhline(self.min_lr, label="min lr", color="black", linestyle ="--")
        plt.xlabel("Training duration")
        plt.title("Learning rate history")
        plt.legend()
        plt.savefig(run_name + "_lr_history.png")


PINN = PINN_types.PINN_MLP.PINN(
    input_dim       = 2,                                                               #x,t 
    output_dim      = 1,                                                               #u
    N_hidden        = 3,                                                               #Number of hidden layers 
    width_layer     = 512,                                                             #Width of hidden layers 
    domains         = [2*torch.pi,1],                                                           #x domain, t domain for non dimensionalization
    PDE_funct       = PINN_utils.Equations.PDE_equations.PDE_convection,                     #PDE residual
    Boundary_funct  = PINN_utils.Equations.Boundary_conditions.BC_equal,               #Boundary residual
    Init_cond_funct = PINN_utils.Equations.Initial_conditions.IC_funct,               #Initial condition residual
    mapping         = None,                                                            #PINN_utils.Mappings.Gaus_Fourier_map(mapping = 100,  sigma =5)
    act             = "wavelet"                                                           #wavelet / tanh
)


PINNTranformer = PINN_types.PINN_Transformer.PINNsformer(
    d_out           = 1,                 
    d_model         = 32,                
    d_hidden        = 512,                  
    N               = 1,
    heads           = 2,
    domains         = [1,1],                                                  #use [1,1] as non-dimensionaliton doesn't work for transformer atm x_pde[:,:,:,:,0]/dim_x would work
    PDE_funct       = PINN_utils.Equations.PDE_equations.PDE_convection,      #PDE residual
    Boundary_funct  = PINN_utils.Equations.Boundary_conditions.BC_equal,      #Boundary residual
    Init_cond_funct = PINN_utils.Equations.Initial_conditions.IC_funct        #Initial condition residual
)



'''

For PINN everything works except sampler_type = transformer (different dims etc)
With training_scheme = causal use sampler_type = static 


For Transformer:

sampler_type = transformer,
training_scheme = classic, 
weight_balance = [NTK2, grad]

'''        


main = Main(model = PINN.to(device),                  #PINN_MLP, PINN_transformer
                 epoch = 150000,                          
                 alpha_lr = 0.90,                     #Learning rate exponential coef 
                 lr_init = 1e-3,                      
                 min_lr  = 5e-4,                      
                 lr_freq = 3500,                      #LR Schedule step freq
                 weight_balance_freq = 10000,         #Weight balance freq 
                 weight_balance     = "Grad",         #None, NTK, NTK2, Grad
                 optimizer          = "Adam",         #Adam , SGD, LBFGS
                 training_scheme    = "Causal",       #Classic, Causal
                 sampler_type       = "Static",       #Dynamic, Static, Transformer (pseudo seq) 
                 sampler_domain     = [2*torch.pi,1], #Domain where to sample [x,t] should be equal to the model domains except for transformer
                 log  =True,                          #Boolean
                 PDE_coefs = [50],                     
                 alpha = 0.9,                         #NTK/NTK2 exponential average coef  
                 epsilon = 6e-6,                      #Causality coefficient , note it has 2500 elements so it must be quite small perhaps modifying it would do some good
                 gradnorm_coefs=[0.5, 1000, 1e-4]     #alpha, sum(w) = T, lr for weights
            )


main.train_PINN()

run_name = "MLP_Adam_Conv_grad_causal_beta=50"

main.plot_log(run_name)





