import numpy as np

import torch
import torch.nn as nn



from functorch import make_functional, vmap, vjp, jvp, jacrev, make_functional_with_buffers



#https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
#It has been modified a bit so that it takes into account the PDE
class NTK:
    def __init__(self, model, PDE_funct,coefs, device, Sampler,alpha =0.9):
        self.model     = model
        self.device    = device
        self.alpha     = alpha
        self.PDE       = PDE_funct
        self.Sampler   = Sampler
        self.coefs     = coefs
        self.domains   = model.domains
    
    def balance(self):
        fnet, params = make_functional(self.model) 
        def fnet_single(params, x):
            return fnet(params, None,x.unsqueeze(0)).squeeze(0)
                
            
        lambda_pde, lambda_bc, lambda_ic = self.KNT_weights(fnet_single, params, size = 25)

        weight_tensor = torch.tensor([lambda_pde, lambda_bc, lambda_ic]).to(self.device)

        for i,j in enumerate(self.model.weights):
            self.model.weights[i] = self.alpha*j+(1-self.alpha)*weight_tensor[i]


    def empirical_ntk_ntk_vps(self, func, params, x1, x2, compute='full', grad = False):
        def get_ntk(x1, x2):
            if grad:                
                def func_x1(params):
                    args = [x1[0].reshape((-1,1)), x1[1].reshape((-1,1))]
                    args.extend(self.coefs)
                    return self.PDE(params,func, self.domains, *args).reshape((-1))                      
                
                def func_x2(params):
                    args = [x2[0].reshape((-1,1)), x2[1].reshape((-1,1))]
                    args.extend(self.coefs)

                    return self.PDE(params, func,self.domains, *args).reshape((-1))  
                
            else:            
                def func_x1(params):
                    return func(params, x1)
        
                def func_x2(params):
                    return func(params, x2)

            output, vjp_fn = vjp(func_x1, params)

            def get_ntk_slice(vec):
                # This computes vec @ J(x2).T
                # `vec` is some unit vector (a single slice of the Identity matrix)
                vjps = vjp_fn(vec)
                # This computes J(X1) @ vjps
                _, jvps = jvp(func_x2, (params,), vjps)
                return jvps

            # Here's our identity matrix
            basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
            return vmap(get_ntk_slice)(basis)
            
        # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
        # Since the x1, x2 inputs to empirical_ntk_ntk_vps are batched,
        # we actually wish to compute the NTK between every pair of data points
        # between {x1} and {x2}. That's what the vmaps here do.
        result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
        
        if compute == 'full':
            return result
        if compute == 'trace':
            return torch.einsum('NMKK->NM', result)
        if compute == 'diagonal':
            return torch.einsum('NMKK->NMK', result)


    
    def KNT_weights(self, fnet_single, params, size = 100):

        points_pde1, points_bc1, points_ic1 = self.Sampler.get_sample() 
        points_pde2, points_bc2, points_ic2 = self.Sampler.get_sample()

        
        x_domain = self.model.domains[0]
        t_domain = self.model.domains[1]



        X_pde1 = torch.cat((points_pde1[0]/x_domain , points_pde1[1]/t_domain), dim = -1)
        X_init1 = torch.cat((points_ic1[0]/x_domain, points_ic1[1]/t_domain), dim = -1)
        X_bound11 = torch.cat((points_bc1[0]/x_domain, points_bc1[1]/t_domain), dim =-1)
        X_bound21 = torch.cat((points_bc1[2]/x_domain, points_bc1[3]/t_domain), dim =-1)
        X_bound1 = torch.cat((X_bound11, X_bound21), dim = 0)
        indexes = torch.randperm(X_bound1.shape[0])
        X_bound1 = X_bound1[indexes]


        X_pde2 = torch.cat((points_pde2[0]/x_domain , points_pde2[1]/t_domain), dim = -1)
        X_init2 = torch.cat((points_ic2[0]/x_domain, points_ic2[1]/t_domain), dim = -1)
        X_bound12 = torch.cat((points_bc2[0]/x_domain, points_bc2[1]/t_domain), dim =-1)
        X_bound22 = torch.cat((points_bc2[2]/x_domain, points_bc2[3]/t_domain), dim =-1)
        X_bound2 = torch.cat((X_bound12, X_bound22), dim = 0)
        indexes = torch.randperm(X_bound2.shape[0])
        X_bound2 = X_bound2[indexes]
        

        trace_pde  = self.empirical_ntk_ntk_vps(fnet_single, params, X_pde1[:size,:]  , X_pde2[:size,:]   , compute="trace", grad = True)    
        trace_bc   = self.empirical_ntk_ntk_vps(fnet_single, params, X_bound1[:size,:], X_bound2[:size,:] , compute="trace")  
        trace_ic   = self.empirical_ntk_ntk_vps(fnet_single, params, X_init1[:size,:] , X_init2[:size,:]  , compute="trace")
    

        TR_pde = np.abs(np.trace(trace_pde.detach().cpu()))
        TR_bc  = np.abs(np.trace(trace_bc.detach().cpu()))
        TR_ic  = np.abs(np.trace(trace_ic.detach().cpu()))

        trace_sum = np.abs(TR_pde) + np.abs(TR_bc) + np.abs(TR_ic)

        return trace_sum/(TR_pde+1e-4),  trace_sum/(TR_bc+1e-4) , trace_sum/(TR_ic+1e-4)

            

#https://github.com/AdityaLab/pinnsformer/blob/main/demo/1d_wave/1d_wave_pinn_ntk.ipynb
class NTK2:
    '''
    Computes it without the PDE res eq 
    Seems to be less efficient than the other NTK
    '''
    def __init__(self, model, device, Sampler,alpha =0.9):
        self.model     = model
        self.device    = device
        self.alpha     = alpha
        self.Sampler   = Sampler

        self.n_params = sum(p.numel() for p in model.parameters())


    def compute_ntk(self,J1, J2):
        Ker = torch.matmul(J1, torch.transpose(J2, 0, 1))
        return Ker

    def balance(self):
        points_pde1, points_bc1, points_ic1 = self.Sampler.get_sample() 

        x_domain = self.model.domains[0]
        t_domain = self.model.domains[1]

        X_pde    = torch.cat((points_pde1[0]/x_domain   , points_pde1[1]/t_domain), dim = -1)
        X_init   = torch.cat((points_ic1[0]/x_domain    , points_ic1[1]/t_domain), dim = -1)
        X_bound1 = torch.cat((points_bc1[0]/x_domain    , points_bc1[1]/t_domain), dim =-1)
        X_bound2 = torch.cat((points_bc1[2]/x_domain    , points_bc1[3]/t_domain), dim =-1)


        pred_pde  = self.model.forward(None, X_pde)
        pred_bc1  = self.model.forward(None, X_bound1)
        pred_bc2  = self.model.forward(None, X_bound2)
        pred_init = self.model.forward(None, X_init)


        J1 = torch.zeros((len(X_pde)   , self.n_params))
        J2 = torch.zeros((len(X_bound1), self.n_params))
        J3 = torch.zeros((len(X_init)  , self.n_params))


        for j in range(len(X_pde)):
            self.model.zero_grad()
            pred_pde[j,0].backward(retain_graph=True)
            J1[j, :] = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

        for j in range(len(X_bound1)):
            self.model.zero_grad()
            pred_bc1[j,0].backward(retain_graph=True)
            pred_bc2[j,0].backward(retain_graph=True)
            J2[j, :] = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

        for j in range(len(X_init)):
            self.model.zero_grad()
            pred_init[j,0].backward(retain_graph=True)
            J3[j, :] = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

        K1 = torch.trace(self.compute_ntk(J1, J1))
        K2 = torch.trace(self.compute_ntk(J2, J2))
        K3 = torch.trace(self.compute_ntk(J3, J3))
    

        K = K1+K2+K3

        w1 = K.item() / K1.item()
        w2 = K.item() / K2.item()
        w3 = K.item() / K3.item()
        
        weight_tensor = torch.tensor([w1, w2, w3]).to(self.device)

        for i,j in enumerate(self.model.weights):
            self.model.weights[i] = self.alpha*j+(1-self.alpha)*weight_tensor[i]


#https://doi.org/10.48550/arXiv.1711.02257
#https://github.com/AvivNavon/AuxiLearn/blob/master/experiments/weight_methods.py
class GradNorm:
    def __init__(self, model,alpha =0.9, T=3, lr = 1e-4):
        
        self.model = model
        self.alpha = alpha
        self.iter = 0
        self.T = T          #sum(w)=T
        self.lr = lr     
        

    def normalize_grad(self, loss):
        if self.iter == 0:
            self.l0 = loss.detach()

        weights = self.model.weights

        optimizer_weights = torch.optim.Adam([weights], lr = self.lr)

        weighted_loss = weights @ loss          

        weighted_loss.backward(retain_graph = True)
        
        Gw = []
        for i in range(0,len(loss)):
            grad = torch.autograd.grad(loss[i]*weights[i], self.model.layers[-1].parameters(),
                        retain_graph=True, create_graph=True, allow_unused=True)[0]
            
            Gw.append(torch.norm(grad))
            
        Gw = torch.stack(Gw)

        loss_ratio = loss.detach()/self.l0      #L_tilde
        rt = loss_ratio / loss_ratio.mean()     #ri

        G_mean =Gw.mean()

        constant = (G_mean*rt**self.alpha).detach()

        L_grad = torch.abs(Gw - constant).sum() #L_grad

        optimizer_weights.zero_grad()

        L_grad.backward()

        optimizer_weights.step()

        weights = weights/weights.sum()*self.T
        
        self.model.weights = torch.nn.Parameter(weights)
        


        

        
