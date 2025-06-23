import torch
from torch.nn.functional import relu, mse_loss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ScaledSigmoid class is use to scale a sigmoid function between a min and a max.
# Here we will use the default value too kept the parameter between 0 and 1
# (because they are probabilities)
class ScaledSigmoid(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        # amplitude = max - min
        self.amplitude = max_val - min_val
        # offset = min
        self.offset = min_val
        # base sigmoïde
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # compute sigmoid in [0,1], then scale to [min_val, max_val]
        return self.sigmoid(x) * self.amplitude + self.offset


# Three_states_analytic_model contain the analytic solution for the Three state model.
# Each gene has it's own model(instance of Three_states_analytic_model) and parameters.
class Three_states_analytic_model(torch.nn.Module):
    def __init__(
        self,
        # Number of cluster to be fitted.
        n_pts,
        
        # Default initial parameter for the parameters.
        l=-3., mu=-3., b=-3., u=-3., ea=-3., eb=-3., d=-3., tc=-3.,
        
        # Theses parameters stop the back propagation for some parameters.
        # It allows to fix a value of a parameter for the whole learning process
        l_grad=True, mu_grad=True, b_grad=True, u_grad=True,
        ea_grad=True, eb_grad=True, d_grad=True, tc_grad=True,
        ):
        super().__init__()
        self.n_pts = n_pts
        self.sig = ScaledSigmoid(min_val=0., max_val=1.)

        # the logits are use for a smoother learning process.
        # the following parameters have a value for each cluster.
        self.logit_l = torch.nn.parameter.Parameter(torch.tensor(torch.ones((self.n_pts))*l), requires_grad=l_grad)
        self.logit_b = torch.nn.parameter.Parameter(torch.tensor(torch.ones((self.n_pts))*b), requires_grad=b_grad)

        # the following parameters have one value by gene.
        self.logit_u = torch.nn.parameter.Parameter(torch.tensor(u), requires_grad=u_grad)
        self.logit_mu = torch.nn.parameter.Parameter(torch.tensor(mu), requires_grad=mu_grad)
        self.logit_ea = torch.nn.parameter.Parameter(torch.tensor(ea), requires_grad=ea_grad)
        self.logit_eb = torch.nn.parameter.Parameter(torch.tensor(eb), requires_grad=eb_grad)
        self.logit_d = torch.nn.parameter.Parameter(torch.tensor(d), requires_grad=d_grad)

    def forward(self):
        # we transform the logit to probabilities with the ScaledSigmoid function.
        self.l = self.sig(self.logit_l)
        self.b = self.sig(self.logit_b)
        self.mu = self.sig(self.logit_mu)
        self.u = self.sig(self.logit_u)
        self.d = self.sig(self.logit_d)
        self.E1 = self.sig(self.logit_ea)
        self.E0 = self.sig(self.logit_eb)

        # We ensure here that the sum of the out probabilties cannot be higher than 1.
        # For that we give the advantage to the parameter mu on b.
        bm = self.b * (1-self.mu)

        # Analytical solution for the fraction of open chromatine.
        frac = (self.l*(bm+self.u))/(self.l*(bm+self.u)+self.mu*self.u)

        # Analytical solution for the mean expression.
        mean = frac * (self.E0 * self.u/(bm+self.u) + self.E1 * bm/(bm+self.u)) / self.d

        # Analytical solution for the numerator of the second moment.
        e2_top = self.l * (
            self.E0**2*self.d**2*self.u +
            self.E0**2*self.d*self.l*self.u +
            self.E0**2*self.d*self.u**2 +
            self.E0**2*self.l*self.u**2 +
            2*self.E0*self.E1*bm*self.d*self.u +
            2*self.E0*self.E1*bm*self.l*self.u +
            self.E1**2*bm**2*self.d +
            self.E1**2*bm**2*self.l +
            self.E1**2*bm*self.d**2 +
            self.E1**2*bm*self.d*self.l +
            self.E1**2*bm*self.d*self.mu
        )

        # Analytical solution for the denominator of the second moment.
        e2_bottom = self.d**2 * (
            bm**2*self.d*self.l +
            bm**2*self.l**2 +
            bm*self.d**2*self.l +
            bm*self.d*self.l**2 +
            bm*self.d*self.l*self.mu +
            2*bm*self.d*self.l*self.u +
            bm*self.d*self.mu*self.u +
            2*bm*self.l**2*self.u +
            2*bm*self.l*self.mu*self.u +
            self.d**2*self.l*self.u +
            self.d**2*self.mu*self.u +
            self.d*self.l**2*self.u +
            2*self.d*self.l*self.mu*self.u +
            self.d*self.l*self.u**2 +
            self.d*self.mu**2*self.u +
            self.d*self.mu*self.u**2 +
            self.l**2*self.u**2 +
            2*self.l*self.mu*self.u**2 +
            self.mu**2*self.u**2
        )

        # Analytical solution for the variance.
        var = e2_top / e2_bottom + mean - torch.pow(mean,2)

        # Analytical solution for a dynamics expression shared by a gene accros different cell type.
        # The dynamics is express by the constant "tc". 
        # In the paper this dynamics is not implemented. 
        # We still need to be sure that this dynamics is in every gene expression.

        # Demonstration :
        # 1/tc = 1/Ton + 1/Toff
        # Ton = 1/u -> 1/Ton = u
        # Toff = 1/bm * (1+mu/l) -> 1/Toff = bm / (1+mu/l)
        # 1/tc = u + bm / (1+mu/l)
        
        tc_inv = self.u + bm/(1+self.mu/self.l)  

        # Output of the model
        return frac, mean, var, tc_inv


# This class take a Three_states_analytic_model like model with the training value to train the model.
class Trainer():
    def __init__(self, model_type, n_pts, lr, **kwargs):
        # Number of cluster to be fit
        self.n_pts = n_pts
        # Learning rate
        self.lr = lr

        # Three_states_analytic_model like class
        self.model = model_type(n_pts, **kwargs)
        # Optimizer for the training step
        self.optimizer = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # Since we log the data for the training, we cannot learn a 0. in the data. for that we had a really small value e.
        self.e = torch.tensor(1e-5)

        # Lists used to keep track of the learning process
        self.loss_frac = []
        self.loss_mean = []
        self.loss_var = []
        self.loss_tc = []
        self.loss_tot = []

        # Initial values used to keep track of the learning process
        self.tot_mse = 0
        self.frac_mse = 0
        self.mean_mse = 0
        self.var_mse = 0
        self.tc_var = 0

    
    def train(
        self,
        log_data_f_, # Logged fracs
        log_data_m_, # Logged means
        log_data_v_, # Logged variances
        N_max=10_000,  # Number of iteration steps
        print_= False, # If True print the loading bar during the training.
        coef_tc=0.0, # Since TC is not in the paper we set the coefficient for the training to 0.
    ):
        import time as t
        self.model.train()
        
        for epoch in range(N_max):
            t0 = t.time()
            
            # Get the values by the model for frac, mean and variance.
            f, m, v, tc_inv = self.model()

            # We compute the Squared Error of the logged data to the computed values. 
            self.frac_mse = torch.nn.functional.mse_loss(torch.log(f+self.e), log_data_f_)
            self.mean_mse = torch.nn.functional.mse_loss(torch.log(m+self.e), log_data_m_)
            self.var_mse = torch.nn.functional.mse_loss(torch.log(v+self.e), log_data_v_)

            # We compute the variance of TC to ensure that it's stay a constant accross the clusters.
            self.tc_var = torch.var(1/tc_inv)

            # Sum of all the losses
            self.tot_mse = self.frac_mse + self.mean_mse + self.var_mse + coef_tc*self.tc_var

            # We keep a track to every individual loss
            self.loss_frac.append(self.frac_mse.item())
            self.loss_mean.append(self.mean_mse.item())
            self.loss_var.append(self.var_mse.item())
            self.loss_tc.append(self.tc_var.item())
            self.loss_tot.append(self.tot_mse.item())
            
            # Learning from loss
            self.optimizer.zero_grad()
            self.tot_mse.backward()
            self.optimizer.step()
            
            if print_:
                print(f'ep: {epoch}, train: {np.round(self.tot_mse.item()*100, 3)} in {np.round(t.time()-t0, 2)}s  ', end='\r')
        if print_:
            print(f'ep: {epoch}, train: {np.round(self.tot_mse.item()*100, 3)} in {np.round(t.time()-t0, 2)}s  ', end='\n')

        # We return the last computed loss
        return self.tot_mse.item()

    # Function to plot the losses.
    def plot(
        self,
        cut=None, # Number of first steps to hide. if None hide the 10 first percent steps. 
        coef_tc=1, # Coef multiplication for TC loss.
    ):
        if cut is None:
            cut = int(len(self.loss_frac)/10)
        
        length = len(self.loss_frac[cut:])
        sns.lineplot(x=range(length), y=self.loss_frac[cut:], label='frac')
        sns.lineplot(x=range(length), y=self.loss_mean[cut:], label='mean')
        sns.lineplot(x=range(length), y=self.loss_var[cut:], label='var')
        sns.lineplot(x=range(length), y=np.array(self.loss_tc[cut:])*coef_tc, label='tc var')
        plt.legend()


# This class is used to contain one Training class for each gene we want to fit
class Fit():
    def __init__(
        self,
        model_type, # Model of analytical solution.
        n_pts, # Number of cluster to fit.
        n_genes, # Number of Genes to fit.
        lr, # Learning rate
        **kwargs, # Here we can set the parameters of the model_type model. But All model / gene will have the same input.
    ):
        # This list contain all Trainer class
        self.trainers = []
        # This list contain all the last loss output of the training process.
        self.losses = []
        # Init the trainer list
        for i in range(n_genes):
            self.trainers.append(Trainer(model_type, n_pts, lr, **kwargs))
    
    # Train all the model.
    def train_all(
        self,
        log_data_f_, # logged frac matrix of shape [N_genes, N_clusters]
        log_data_m_, # logged mean matrix of shape [N_genes, N_clusters]
        log_data_v_, # logged variance matrix of shape [N_genes, N_clusters]
        **kwargs,
    ):
        for i in tqdm(range(log_data_f_.shape[0])):
            self.losses.append(
                self.trainers[i].train(log_data_f_[i], log_data_m_[i], log_data_v_[i], **kwargs)
            )
