import math
import torch
from typing import Union
from torch import Tensor
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.integration import integrate_TLM
import numpy as np
import numpy.testing as npt
from svise import odes
from scipy.integrate import solve_ivp
from torch.autograd.functional import jacobian
from SviKalmanNet.model import train_model_config
from SviKalmanNet.dnn_net import DNN_SKalmanNet
from SviKalmanNet.odes import Series_RLC_Circuit
from experiment2.get_experiment import get_experiment
import torchsde
def series_RLC(t, x, R=100, L=5, C=0.1):
    v_C = x[..., 0]  # V_t
    i_L = x[..., 1]  # I_t
    i_source = 2 * math.sin((2 * math.pi / 3) * t)
    dx = (1 / C) * i_L
    dy = (-1 / (L * C)) * v_C - (R / L) * i_L + (1 / L) * i_source
    return [dx, dy]


def chua_circuit(t, x):
    R = 1600
    C1 = 10
    C2 = 100
    L = 20e-3
    u1, u2 = x[...,0], x[...,1]
    m0 = -1.27
    m1 = -0.75
    Bp = 1.0
    #  g(u1)
    g_u1 = m1 * u1 + 0.5 * (m0 - m1) * (Bp)

    # 微分方程
    du1_dt = (u2 / (R * C1)) - (u1 / (R * C1)) - (g_u1 / C1)
    du2_dt = (u1 / (R * C2)) - (u2 / (R * C2)) - (u2 / C2)
    dIL_dt = -u2 / L

    return [du1_dt, du2_dt]


class Particle_Filter():

    def __init__(self, ode_model: train_model_config):
        self.model_name = ode_model.name
        self.model = ode_model
        self.x_dim = ode_model.x_dim
        self.y_dim = ode_model.y_dim
        self.x0 = ode_model.x0.reshape(1, self.x_dim)  # 1X2
        self.state_history = self.x0.detach().clone()
        self.reset(clean_history=True)
        #
        # self.G = torch.eye(self.y_dim)

        self.G = ode_model.h
        # self.Q_diag = (torch.ones(self.x_dim)*1e-2)
        self.Q_diag = ode_model.Q_diag

        self.var = ode_model.var  #### measurement

        # npt.assert_array_equal(self.G, torch.eye(self.x_dim))

        # test_ode = lambda t, x:  series_RLC(t, x)   # experiment1
        test_ode = lambda t, x: chua_circuit(t, x)  # experiment3
        self.exp_ode = test_ode
        self.np_ode = lambda x: np.stack(self.exp_ode(0.0, x), -1)
        self.torch_ode = lambda x: torch.stack(self.exp_ode(0.0, x), -1)

    def reset(self, clean_history=False):
        self.state_post = self.x0.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_post), dim=0)
        if clean_history:
            self.state_history = self.x0.detach().clone()

    def d2x_dtdx(self, x):
        """Jacobian of x (d-dim vector)"""
        xt = torch.tensor(x)
        jac = jacobian(self.torch_ode, xt)
        return jac.numpy()

    def dstep_dx(self, x, t, dt):
        """Compute resolvent (propagator) of the TLM. I.e. the Jacobian of `step(x)`."""
        # return integrate_TLM(d2x_dtdx(x), dt, method="approx")
        return integrate_TLM(self.d2x_dtdx(x), dt, method="approx")  # forward euler

    def filtering(self, data_t, data_y, data_true):
        yy = data_y.numpy()     # N X dim ###########注意输入
        # xx = data_true.numpy()
        Ny, num_states = yy.shape

        ############################# Chua circuit / experiment3 ###########################
        dt = 0.02  #
        K = 1000
        Ko = Ny - 1
        dko = int(
            np.ceil(K / (Ko + 1))
        )  # ratio of number of simulation points to observation points
        # dko = 2  #
        dto = dt
        ############################# RCL / experimrnt1 #############################
        # K = 10000
        # Ko = Ny - 1
        # dko = int(
        #     np.ceil(K / (Ko + 1))
        # )  # ratio of number of simulation points to observation points
        # dto = (data_t[1] - data_t[0]).item()

        tseq = modelling.Chronology(Ko=Ko, dko=dko, dto=dto)
        step = modelling.with_rk4(self.np_ode, autonom=True, stages=1)  # euler-maruyama instead
        Q = modelling.CovMat(self.Q_diag.numpy(), kind="diag")

        Dyn = {
            "M": 2,
            "model": step,
            "linear": self.dstep_dx,
            "noise": modelling.GaussRV(C=Q),
        }
        # we want the initial condition to be almost useless
        y_true = data_true
        data_range = (y_true.max(0).values - y_true.min(0).values) / 2
        # that dynamics diverge (hopf bifurcation is quite difficult)
        var_init = 0.001*data_range  ##################################### 初值设置

        P0 = modelling.CovMat(var_init, kind="diag")
        tempt_x0 = self.x0.squeeze()
        # sol_x0 = torchsde.sdeint(Series_RLC_Circuit(), self.x0, (tseq.tt[0], tseq.tto[0]), method='euler')
        sol_x0 = solve_ivp(
            self.exp_ode, (tseq.tto[0], tseq.tt[0]), tempt_x0, atol=1e-9, rtol=1e-6
        )
        x0 = sol_x0.y.T[0]  # working backwrads to find the true initial condition

        assert not np.isnan(x0).any(), "x0 contains NaN after solve_ivp"

        X0 = modelling.GaussRV(C=P0, mu=x0)
        # # setting up observation model
        jj = np.arange(num_states)  # obs_inds
        # Obs = modelling.Id_Obs(num_states)
        Obs = modelling.partial_Id_Obs(num_states, jj)

        R = modelling.CovMat(self.var.squeeze().numpy(), kind="diag")
        Obs["noise"] = modelling.GaussRV(C=R)


        # setting up HMM
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
        xx, __ = HMM.simulate()

        infl = 0.02
        da_method = {
            "PartFilt": da.PartFilt(N=2000, reg=0.3, NER=0.1),  # we only use the particle filter
            # "ExtKF": da.ExtKF(1.00),
        }
        da_data = {}
        for name, method in da_method.items():
            print(f"Running: {name}")
            HMM_tmp = HMM.copy()

            if name == "iEnKS":
                # iEnKS doesn't work with set diffusion
                HMM_tmp.Dyn.noise.C = 0.0
            method.assimilate(HMM_tmp, xx, yy)
            try:  # check for smoothed soln
                mu = method.stats.mu.s  # smoothed
                std = method.stats.spread.s
                smoothed = True
            except AttributeError:  # else use recursive soln
                print(f"{name} doesn't have a smoothed solution")
                mu = method.stats.mu.a
                std = method.stats.spread.a
                smoothed = False
            if np.isnan(mu).any() or np.isnan(std).any():
                print("nans detected, suggest rerun.")
            da_data[f"{name}_mu"] = mu
            da_data[f"{name}_std"] = std
            da_data[f"{name}_t"] = tseq.tto - dto
            da_data[f"{name}_s"] = smoothed
        x_post = torch.tensor(mu)        #dim X N
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=0)
        self.est_var = torch.tensor(std **2)

class Particle_Filter2():

    def __init__(self, ode_model: train_model_config):
        self.model_name = ode_model.name
        self.model = ode_model
        self.x_dim = ode_model.x_dim
        self.y_dim = ode_model.y_dim
        self.x0 = ode_model.x0.reshape(1,self.x_dim)  # 1X2
        self.state_history = self.x0.detach().clone()
        self.reset(clean_history=True)

        self.G = ode_model.G

        self.Q_diag = ode_model.Q_diag
        self.var = ode_model.var   #### measurement
        # npt.assert_array_equal(self.G, torch.eye(self.x_dim))
        _, self.exp_ode, _, self.tf, self.num_x0, noise, _, self.alpha, self.beta = get_experiment(self.model_name)
        self.np_ode = lambda x: np.stack(self.exp_ode(0.0, x), -1)
        self.torch_ode = lambda x: torch.stack(self.exp_ode(0.0, x), -1)

    def reset(self, clean_history=False):
        self.state_post = self.x0.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_post), dim=0)
        if clean_history:
            self.state_history = self.x0.detach().clone()

    def d2x_dtdx(self, x):
        """Jacobian of x (d-dim vector)"""
        xt = torch.tensor(x)
        jac = jacobian(self.torch_ode, xt)
        return jac.numpy()

    def dstep_dx(self, x, t, dt):
        """Compute resolvent (propagator) of the TLM. I.e. the Jacobian of `step(x)`."""
        # return integrate_TLM(d2x_dtdx(x), dt, method="approx")
        return integrate_TLM(self.d2x_dtdx(x), dt, method="approx")  # forward euler

    def corruption(self, t, x, dx, alpha: float, beta: float):
        dx_mod = dx
        dx_mod[..., 0] = dx[..., 0] - alpha * x[..., 1] + beta
        return dx_mod

    def filtering(self, data_t, data_y, data_true):
        yy = data_y.numpy()  # N X dim
        xx = data_true.numpy()
        Ny, num_states = yy.shape
        K = 1000
        Ko = Ny - 1
        dko = int(
            np.ceil(K / (Ko + 1))
        )  # ratio of number of simulation points to observation points
        dto = (data_t[1] - data_t[0]).item()
        tseq = modelling.Chronology(Ko=Ko, dko=dko, dto=dto)
        step = modelling.with_rk4(self.np_ode, autonom=True, stages=2)  # euler-maruyama instead
        # I think this is used every time step based on HMM.tseq.dt (i.e. dynamic update rate)
        # can check this by stepping through the extkf solution
        Q = modelling.CovMat(self.Q_diag.numpy(), kind="diag")
        # Q = modelling.CovMat(np.cov(xx.T)) * 1e-2
        Dyn = {
            "M": 2,
            "model": step,
            "linear": self.dstep_dx,
            "noise": modelling.GaussRV(C=Q),
        }
        # we want the initial condition to be almost useless
        y_true = data_true
        data_range = (y_true.max(0).values - y_true.min(0).values) / 2
        # chosen so that iEnKS does not sample such
        var_init = 0.001 * torch.ones(2)
        P0 = modelling.CovMat(var_init, kind="diag")
        # initial condition for hopf
        if self.model_name == "Hopf bifurcation":
            x0 = self.x0.squeeze(0)
        else:
            alpha = self.alpha
            beta = self.beta
            # true_ode = lambda t, x: self.corruption(t, x, self.exp_ode(t, x), alpha, beta)
            sol_x0 = solve_ivp(
                self.exp_ode, (tseq.tto[0], tseq.tt[0]), self.num_x0, atol=1e-9, rtol=1e-6
            )
            x0 = sol_x0.y.T[-1]  # working backwrads to find the true initial condition
        X0 = modelling.GaussRV(C=P0, mu=x0)
        # setting up observation model
        jj = np.arange(num_states)  # obs_inds
        Obs = modelling.partial_Id_Obs(num_states, jj)
        R = modelling.CovMat(self.var.numpy(), kind="diag")
        Obs["noise"] = modelling.GaussRV(C=R)

        # setting up HMM
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
        xx, __ = HMM.simulate()
        # creating list of da methods
        # these are some sensible parameters, by no means optimal in all cases
        infl = (data_range.numpy() * 0.01) ** 2
        da_method = {
            "PartFilt": da.PartFilt(N=2000, reg=0.5, NER=0.1),  # we only ended up using the PF
            # "ExtKF": da.ExtKF(1.00),
        }
        da_data = {}
        for name, method in da_method.items():
            print(f"Running: {name}")
            HMM_tmp = HMM.copy()
            if name == "iEnKS":
                # iEnKS doesn't work with set diffusion
                HMM_tmp.Dyn.noise.C = 0.0
            method.assimilate(HMM_tmp, xx, yy)
            try:  # check for smoothed soln
                mu = method.stats.mu.s  # smoothed
                std = method.stats.spread.s
                smoothed = True
            except AttributeError:  # else use recursive soln
                print(f"{name} doesn't have a smoothed solution")
                mu = method.stats.mu.a
                std = method.stats.spread.a
                smoothed = False
            if np.isnan(mu).any() or np.isnan(std).any():
                print("nans detected, suggest rerun.")
            da_data[f"{name}_mu"] = mu
            da_data[f"{name}_std"] = std
            da_data[f"{name}_t"] = tseq.tto - dto
            da_data[f"{name}_s"] = smoothed

        x_post = torch.tensor(mu)  # dim X N
        self.est_var = torch.tensor(std**2)
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=0)


class SVI_KalmanNet_Filter:
    def __init__(self, model: train_model_config):
        self.model_name = model.name
        self.GSSModel = model
        self.x_dim = model.x_dim
        self.y_dim = model.y_dim
        self.init_state = model.x0   # 2X1
        self.var = model.var   #1X2
        self.init_P = 1e-3*torch.ones_like(self.init_state)

        # self.data_t = torch.lo_diagad(data_path + 'time.pt')
        # self.data_y = torch.load(data_path + 'obs.pt')
        # self.prestate = torch.load(data_path + 'prestate.pt')
        # self.precov = torch.load(data_path + 'precov.pt')

        self.kf_net = DNN_SKalmanNet(self.x_dim, self.y_dim)

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        self.P_post = self.init_P.detach().clone()

        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.state_history2 = self.init_P.detach().clone()

        self.state_history = torch.cat((self.state_history, self.state_post), dim=-1)
        self.state_history2 = torch.cat((self.state_history2, self.P_post), dim=-1)

    def filtering(self, prestate, cov, obs):
        # observation: column vector
        # 输入都是列向量

        if self.dnn_first:
            self.state_post_past = self.init_state.detach().clone()
            self.obs_past = obs.detach().clone()

        ## input 1: x_{k| k-1} - x_{k-1 | k-1}
        # state_inno = self.state_post - prestate
        state_inno = prestate

        ## input 2: residual
        # y_predict = torch.matmul(self.GSSModel.h(), prestate)
        y_predict = self.GSSModel.h() @ prestate
        residual = obs - y_predict

        ## input 3: y_k - y_{k-1}
        diff_obs = obs - self.obs_past
        precov = cov


        # outer_product = torch.matmul(residual, residual.t())  # 2x2

        meas_cov = self.var.reshape(-1, 1)

        ## input 4: linearization error
        H_jacob = self.GSSModel.Jacobian_h(prestate)
        linearization_error = y_predict - H_jacob @ prestate

        # H_jacob_in = F.normalize(H_jacob_in, p=2, dim=0, eps=1e-12)
        # linearization_error_in = linearization_error

        # linearization_error_in = F.normalize(linearization_error, p=2, dim=0, eps=1e-12)
        x_post, cov_post = self.kf_net(state_inno, precov, residual, meas_cov)

        # Pk = torch.diag(cov)
        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # (Pk, Sk) = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in, linearization_error, H_jacob.reshape((-1,1)))

        # K_gain = torch.transpose(H_jacob, 0, 1) @ Sk
        #
        # x_post = prestate + Pk @ K_gain @ residual

        self.dnn_first = False
        # self.state_post_past = prestate.detach().clone()
        # self.obs_past = obs.detach().clone()
        self.state_post = x_post.detach().clone()
        self.P_post = cov_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=-1)
        self.state_history2 = torch.cat((self.state_history2, cov_post.clone()), dim=-1)