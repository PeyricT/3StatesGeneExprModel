import torch
from torch.nn.functional import relu, mse_loss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ScaledSigmoid(torch.nn.Module):
    def __init__(self, min_val=1.0, max_val=3.0):
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


class Three_states_analytic_model_tc(torch.nn.Module):
    def __init__(
        self, n_pts,
        l=-3., mu=-3., b=-3., u=-3., e=-3., d=-3., tc=-3.,
        l_grad=True, mu_grad=True, b_grad=True, u_grad=True,
        e_grad=True, d_grad=True, tc_grad=True,
        ):
        super().__init__()
        self.n_pts = n_pts
        self.sig = ScaledSigmoid(min_val=0., max_val=1.)

        self.logit_l = torch.nn.parameter.Parameter(torch.tensor(torch.ones((self.n_pts))*l), requires_grad=l_grad)
        self.logit_b = torch.nn.parameter.Parameter(torch.tensor(torch.ones((self.n_pts))*b), requires_grad=b_grad)
        self.logit_u = torch.nn.parameter.Parameter(torch.tensor(torch.ones((self.n_pts))*u), requires_grad=u_grad)
        self.logit_mu = torch.nn.parameter.Parameter(torch.tensor(mu), requires_grad=mu_grad)
        self.logit_e = torch.nn.parameter.Parameter(torch.tensor(e), requires_grad=e_grad)
        self.logit_d = torch.nn.parameter.Parameter(torch.tensor(d), requires_grad=d_grad)

    def forward(self):
        self.l = self.sig(self.logit_l)
        self.b = self.sig(self.logit_b)
        self.mu = self.sig(self.logit_mu)
        self.u = self.sig(self.logit_u)
        self.d = self.sig(self.logit_d)
        self.E1 = self.sig(self.logit_e)
        self.E0 = 0

        bm = self.b * (1-self.mu)
        frac = (self.l*(bm+self.u))/(self.l*(bm+self.u)+self.mu*self.u)
        mean = frac * (self.E0 * self.u/(bm+self.u) + self.E1 * bm/(bm+self.u)) / self.d
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
        var = e2_top / e2_bottom + mean - torch.pow(mean,2)

        #1/tc = 1/Ton + 1/Toff
        # Ton = 1/u -> 1/Ton = u
        # Toff = 1/bm * (1+mu/l) -> 1/Toff = bm / (1+mu/l)
        # 1/tc = u + bm / (1+mu/l)
        tc_inv = self.u + bm/(1+self.mu/self.l)  
        
        return frac, mean, var, tc_inv


class Trainer_tc():
    def __init__(self, model_type, n_pts, lr, **kwargs):
        self.n_pts = n_pts
        self.lr = lr
        
        self.model = model_type(n_pts, **kwargs)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.model.parameters())

        self.e = torch.tensor(1e-5)
        self.loss_frac = []
        self.loss_mean = []
        self.loss_var = []
        self.loss_tc = []
        self.loss_tot = []
        self.tot_mse = 0
        self.frac_mse = 0
        self.mean_mse = 0
        self.var_mse = 0
        self.tc_var = 0

    def train(self, log_data_f_, log_data_m_, log_data_v_, N_max=10_000, N_min=1000, print_=False, coef_tc=0.1):
        import time as t
        self.model.train()
        for epoch in range(N_max):
            t0 = t.time()
            f, m, v, tc_inv = self.model()
        
            self.frac_mse = torch.nn.functional.mse_loss(torch.log(f+self.e), log_data_f_)
            self.mean_mse = torch.nn.functional.mse_loss(torch.log(m+self.e), log_data_m_)
            self.var_mse = torch.nn.functional.mse_loss(torch.log(v+self.e), log_data_v_)
            self.tc_var = torch.var(1/tc_inv)
            self.tot_mse = self.frac_mse + self.mean_mse + self.var_mse + coef_tc*self.tc_var
        
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
            
        return self.tot_mse.item()

    def plot(self, cut=None, coef_tc=1):
        if cut is None:
            cut = int(len(self.loss_frac)/10)
        
        length = len(self.loss_frac[cut:])
        sns.lineplot(x=range(length), y=self.loss_frac[cut:], label='frac')
        sns.lineplot(x=range(length), y=self.loss_mean[cut:], label='mean')
        sns.lineplot(x=range(length), y=self.loss_var[cut:], label='var')
        sns.lineplot(x=range(length), y=np.array(self.loss_tc[cut:])*coef_tc, label='tc var')
        plt.legend()


class Fit_tc():
    def __init__(self, model_type, n_pts, n_genes, lr, **kwargs):

        self.trainers = []
        self.losses = []
        for i in range(n_genes):
            self.trainers.append(Trainer_tc(model_type, n_pts, lr, **kwargs))

    def train_all(self, log_data_f_, log_data_m_, log_data_v_, **kwargs):
        for i in tqdm(range(log_data_f_.shape[0])):
            self.losses.append(
                self.trainers[i].train(log_data_f_[i], log_data_m_[i], log_data_v_[i], **kwargs)
            )
