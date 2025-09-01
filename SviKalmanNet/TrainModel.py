import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from tqdm import tqdm
from svise import sde_learning, sdeint
import torch.optim.lr_scheduler as lr_scheduler
from svise.sde_learning import *

num_iters = 2000
loss_fn = torch.nn.MSELoss()
class TrainModel:
    def __init__(self, ode_model, data_path):
        train_data = torch.load(data_path)
        self.data_t = train_data["time"]
        self.data_x = train_data["state"]
        self.data_y = train_data["obs"]

        self.valid_t = train_data["time"]
        self.valid_x = train_data["state"]

        self.d = self.data_x.shape[1]

        self.G = ode_model.G
        self.var = ode_model.var

        #self.valid_loss_svi = []
        print("number data")
        print(len(self.data_t))
        print(self.data_x.shape[0])
        self.data_idx = 0
        self.train_count = 0
        self.batch_model = 1
        self.trajac = ode_model.num_traj
        t_span = (self.data_t.min(), self.data_t.max())
        print("t_span")
        print(t_span)
        self.loss_best = 1e4
        self.model = sde_learning.SparsePolynomialSDE(
             d=self.d,
             t_span=t_span,
             degree=5,  # degree of polynomial terms
             n_reparam_samples=32,
             G=self.G,
             num_meas=self.G.shape[0],
             measurement_noise=self.var,
             train_t=self.data_t,
             train_x=self.data_y,
             input_labels=["x", "y"],
             n_quad= 120,
             n_tau = 110,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-1)
        batch_size = min(len(self.data_t), 128)
        self.train_loader = DataLoader(TensorDataset(self.data_t, self.data_y), batch_size=batch_size, shuffle=True)
        self.num_data = len(self.data_t)
        # sparse learning takes a long time to converge
        self.model.train()
        self.num_epochs = num_iters // len(self.train_loader)
        self.transition_iters = self.num_epochs // 2
        assert self.transition_iters < num_iters
        # self.summary_freq = 1000
        # lr = 1e-1
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-1)
        # self.optimizer = torch.optim.Adam(
        #      [
        #          {"params": self.model.state_params()},
        #          {"params": self.model.sde_params(), "lr": 1e-2},
        #      ],
        #      lr=lr,
        # )
        # self.batch_size = int(min(len(self.data_t), 128))
        # self.train_loader = DataLoader(TensorDataset(self.data_t, self.data_y),
        #                                    batch_size= self.batch_size, shuffle=True)
        # self.num_epochs = num_iters // len(self.train_loader)

    def train_batch_model(self):
        beta = min(1.0, self.train_count / self.transition_iters)
        for tbatch, ybatch in self.train_loader:
            self.optimizer.zero_grad()
            loss = -self.model.elbo(tbatch, ybatch, beta=beta, N=self.num_data)
            loss.backward()
            self.optimizer.step()
        if self.train_count % 20 == 0:
            mu = self.model.marginal_sde.mean(self.valid_t)
            var = self.model.marginal_sde.K(self.valid_t).diagonal(dim1=-2, dim2=-1)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(self.valid_x, mu)
            # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
            loss_dB = 10 * torch.log10(loss)
            print(f'loss [dB] = {loss_dB:.4f}')

            # var_names = ["x", "y"]
            # print("##################################################\n")
            # print("The learned system model are expressed as\n")
            # for j, eq in enumerate(self.model.sde_prior.feature_names):
            #     print(f"d{var_names[j]} = {eq}")

            os.makedirs(os.path.dirname('./.model_saved/'), exist_ok=True)
            self.save_path = './.model_saved/'

            torch.save(self.model,
                       self.save_path + '(PRLCC) SVI_' + str(self.train_count) + '.pt')


    def validate(self, tester):
        if tester.loss.item() < self.loss_best:
            try:
                torch.save(tester.filter.kf_net, self.save_path + '(PRLCC) SVI_' + '_best.pt')
                print(f'Save best model at {self.save_path} & train {self.train_count} & loss [dB] = {tester.loss:.4f}')
                self.loss_best = tester.loss.item()
            except:
                pass
        self.valid_loss = tester.loss.item()