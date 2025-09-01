import torch
from SviKalmanNet.filter import SVI_KalmanNet_Filter, Particle_Filter2,  Particle_Filter
import time
import os
from experiment.NRMSE import rmse, rmse2
from datetime import timedelta
from svise import odes

class Tester():
    def __init__(self, filter, data_path, svi_path, kalman_model, is_validation=False):
        # Example:
        #   data_path  = './.data/syntheticNL/test/(true)
        #   sys_model = './.model_saved/(PRLCC) SVI_' + str(epoch) + '.pt'
        #   model_path = ./.model_saved/(PRLCC) Svi_KalmanNet_self.count.pt'
        # self.sys_model = torch.load(svi_path)
        self.sys_model = torch.load(svi_path)


        self.filter = filter
        self.filter.kf_net = torch.load(kalman_model)
        self.filter.kf_net.initialize_hidden()

        self.loss_fn = torch.nn.MSELoss()

        data_t = torch.load(data_path + 'time.pt').to(dtype=torch.float64)   # traj X num
        data_y = torch.load(data_path + 'obs.pt').to(dtype=torch.float64)   # traj X num X 2
        data_true = torch.load(data_path + 'true_state.pt').to(dtype=torch.float64)
        ####################################

        # self.sys_model.eval()
        # self.sys_model.sde_prior.reset_sparse_index()   # 重置索引
        # self.sys_model.sde_prior.update_sparse_index()  # 更新索引

        self.num_traj = data_true.shape[0]
        self.num_data = data_true.shape[1]
        self.x_dim = data_true.shape[2]
        mu = torch.zeros(self.num_traj, self.num_data, self.x_dim)
        var = torch.zeros(self.num_traj, self.num_data, self.x_dim)
        for i in range(self.num_traj):
            mu[i, :, :] = self.sys_model.marginal_sde.mean(data_t[i])
            var[i, :, :] = self.sys_model.marginal_sde.K(data_t[i]).diagonal(dim1=-2, dim2=-1)
        x_hat = torch.zeros_like(data_true)
        P_hat = torch.zeros_like(data_true)
        start_time = time.monotonic()
        with torch.no_grad():
            for i in range(self.num_traj):
                self.filter.state_post = mu[i, 0, :].reshape((-1, 1))
                if isinstance(filter, SVI_KalmanNet_Filter):
                    for ii in range(1, self.num_data):
                        self.filter.filtering(mu[i, ii, :].reshape((-1, 1)),
                                              var[i, ii, :].reshape(-1, 1),
                                              data_y[i, ii, :].reshape((-1, 1)),
                                             )
                x_hat[i,:,:] = self.filter.state_history[:, -self.num_data:].t()
                P_hat[i, :, :] = self.filter.state_history2[:, -self.num_data:].t()
                self.filter.reset(clean_history=False)
            loss = rmse2(data_true[0, 1:, :], x_hat[0, 1:, :])
            self.loss = 10 * torch.log10(loss)
            loss2 = rmse(data_true[:, 1:, :], x_hat[:, 1:, :])
            self.loss2 = 10 * torch.log10(loss2)
            print(f'loss [dB] = {self.loss2.item():.4f}')

            if not is_validation:
                self.est_mean = x_hat.squeeze(0)
                self.est_var = P_hat.squeeze(0)

            # Compute loss at instantaneous time
            # self.loss_instant = torch.zeros(self.data_true[:, 1:, :].shape[-1])
            # for i in range(self.data_true[:, 1:, :].shape[-1]):
            #     self.loss_instant[i] = self.loss_fn(self.data_true[:, i+1, :], x_hat[:, i+1, :])
            # self.loss_instant_dB = 10 * torch.log10(self.loss_instant)


class Tester_PF:
    def __init__(self, filter: [Particle_Filter, Particle_Filter2], data_path):
        self.filter = filter
        self.data_t = torch.load(data_path + 'time.pt').to(dtype=torch.float64)   # traj X num
        self.data_y = torch.load(data_path + 'obs.pt').to(dtype=torch.float64)    # traj  X num X 2
        self.data_true = torch.load(data_path + 'true_state.pt').to(dtype=torch.float64)
        self.data_num = self.data_y.shape[0]
        self.seq_len = self.data_y.shape[1]
        self.loss_fn = torch.nn.MSELoss()

        x_hat = torch.zeros_like(self.data_true)    # traj  X num X 2
        P_hat = torch.zeros_like(self.data_true)
        start_time = time.monotonic()
        with torch.no_grad():
            for i in range(self.data_num):
                self.filter.state_post = self.data_true[i, 0, :]     # traj X 2 X num
                # if isinstance(filter, Particle_Filter):
                self.filter.filtering(self.data_t[i],
                                      self.data_y[i],
                                      self.data_true[i])

                x_hat[i] = self.filter.state_history[-self.seq_len:,:]
                P_hat[i] = self.filter.est_var
                self.filter.reset(clean_history=False)
        self.est_mean = x_hat.squeeze(0)
        self.est_var = P_hat.squeeze(0)
        loss = rmse2(self.data_true[0, 1:, :], x_hat[0, 1:, :])
        self.loss = 10 * torch.log10(loss)
        # print(f'loss_PF [dB] = {loss_dB.item():.4f}')
        loss2 = rmse(self.data_true[:, 1:, :], x_hat[:, 1:, :])
        self.loss2 = 10 * torch.log10(loss2)
        print(f'loss [dB] = {self.loss2.item():.4f}')

class Tester_for_svi:
    def __init__(self, data_path, svi_path, is_validation=False, j=None):
        # data_path = './data/PRLCC/valid_data/PRLCC_permille_0100data_01.pt'
        self.loss_fn = torch.nn.MSELoss()
        if is_validation:
            self.test_model = svi_path
        else:
            self.test_model = torch.load(svi_path)
        self.data_t = torch.load(data_path + "time.pt").to(dtype=torch.float64)
        self.data_y = torch.load(data_path + "obs.pt").to(dtype=torch.float64)
        self.data_true = torch.load(data_path + "true_state.pt").to(dtype=torch.float64)
        with torch.no_grad():
            if is_validation:
                num_traj = self.data_y.shape[0]
                num_data = self.data_y.shape[1]
                d = self.data_true.shape[2]
                mu = torch.zeros(num_traj, num_data, d)
                var = torch.zeros(num_traj, num_data, d)
                for i in range(num_traj):
                    valid_time = self.data_t[i]
                    mu[i,:,:] = self.test_model.marginal_sde.mean(valid_time)
                    var[i,:,:] = self.test_model.marginal_sde.K(valid_time).diagonal(dim1=-2, dim2=-1)

                loss = rmse(self.data_true[:,1:,:], mu[:, 1:, :])
                # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
                self.loss_dB = 10 * torch.log10(loss)
                print(f'loss [dB] = {self.loss_dB:.4f}')
                save_path = './.model_saved/'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(self.test_model, save_path + '(PRLCC) SVI_' + str(j) + '.pt')
                if loss.item() < 1e-4:
                    try:
                        torch.save(self.test_model, './.model_saved/SVI_best.pt')
                        print(
                            f'Save best model at ./.model_saved/ & train {str(j)} & loss [dB] = {self.loss_dB:.4f}')
                    except:
                        pass
            else:
                self.test_model.eval()
                num_traj = self.data_y.shape[0]
                num_data = self.data_y.shape[1]
                d = self.data_true.shape[2]
                mu = torch.zeros(num_traj, num_data, d)
                var = torch.zeros(num_traj, num_data, d)
                for i in range(num_traj):
                    valid_time = self.data_t[i]
                    mu[i, :, :] = self.test_model.marginal_sde.mean(valid_time)
                    var[i, :, :] = self.test_model.marginal_sde.K(valid_time).diagonal(dim1=-2, dim2=-1)
                loss = rmse2(self.data_true[0, 1:, :], mu[0, 1:, :])
                self.loss = 10 * torch.log10(loss)
                loss2 = rmse(self.data_true[:, 1:, :], mu[:, 1:, :])
                self.loss2 = 10 * torch.log10(loss2)
                # print(f'loss [dB] = {self.loss_dB:.4f}')
                self.est_mean = mu.squeeze(0)
                self.est_var = var.squeeze(0)


########################################################################################
from .GM_UKF import RobustGMUKF_ODE
import numpy as np
from functools import partial
class Tester_GM_UKF:
    def __init__(self, model_name, data_path):
        system_name = model_name
        if system_name == "Damped linear oscillator":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="linear")
            self.tf = 20.0
            self.x0 = [2.5, -5.0]  # these intial conditions make for better eq learning
            noise = 1e-1

        elif system_name == "Damped cubic oscillator":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="cubic")
            self.tf = 15.0
            self.x0 = [0.0, -1.0]  # these intial conditions make for better eq learning
            self.noise = 1e-2
        elif system_name == "Selkov glycolysis model":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.selkov(t, x)
            self.tf = 30.0
            self.x0 = [0.7, 1.25]
            self.noise = 1e-2

        data = torch.load(data_path)
        system_choice = data["name"]
        d = data["d"]
        self.num_data = data["num_data"]
        self.t = data["t"]
        self.H= data["G"]
        self.y_data = data["y_data"]  # 1 X seq_len X N
        self.true_data= data["y_true"]
        var = data["var"]
        noise_percent = data["noise_percent"]
        self.x0 = data["x0"]

        degree = data["degree"]
        Q = data["pnoise_cov"]
        self.n_obs = self.H.shape[0]
        self.device = "cpu"
        self.dtype = torch.float64
        self.Q = torch.diag(torch.tensor(Q, dtype=self.dtype))
        self.R = torch.diag(torch.tensor(var, dtype=self.dtype))

    def forward(self):
        def f_lv(x, t=0.0):
            X = self.test_ode(t, x)
            return np.array(X, dtype=np.float64)
            # 观测：线性取样 + 噪声（可替换为你的 h）
        def h_lv(x, dt=None):
            y = self.H @ x
            return y.detach().cpu().numpy().astype(np.float64)

        # 构造滤波器：连续微分方程 + RK4
        gmukf = RobustGMUKF_ODE(
            n_states=self.num_states,
            n_obs=self.n_obs,
            f=f_lv,  # 连续系统：dx/dt = f(x,t)
            h=h_lv,
            Q=self.Q,
            R=self.R,
            f_mode='dx',  # 'dx' 表示 f 返回 dx/dt
            integrator='rk4',
            kappa=0,
            alpha=1e-3,
            beta=2,
            huber_c=3.345,
            ps_window=1,
            ps_enable=True,
            ps_z_norm=True,
            irls_iters=2,
            device=self.device,
            dtype=self.dtype
        )
        Y = self.y_data
        batch = Y.shape[0]
        dt = self.t[:,1]-self.t[:,0]
        dt = np.array(dt, dtype=np.float64)

        X0 = torch.as_tensor(self.x0, dtype=torch.float64).unsqueeze(0)
        P0 = 1e-3 * torch.eye(self.num_states, dtype=self.dtype).unsqueeze(0).repeat(batch, 1, 1)

        X_est, P_est = gmukf.run(X0, P0, Y, dt=dt)

        loss = rmse2(self.true_data[0, 1:, :], X_est[0, 1:, :])
        self.loss = 10 * torch.log10(loss)

        return X_est.detach().cpu().numpy(), P_est.detach().cpu().numpy(), self.loss.detach().cpu().numpy()

class Tester_GM_UKF:
    def __init__(self, model_name, data_path):
        system_name = model_name
        if system_name == "Damped linear oscillator":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="linear")
            self.tf = 20.0
            self.x0 = [2.5, -5.0]  # these intial conditions make for better eq learning
            noise = 1e-1

        elif system_name == "Damped cubic oscillator":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="cubic")
            self.tf = 15.0
            self.x0 = [0.0, -1.0]  # these intial conditions make for better eq learning
            self.noise = 1e-2
        elif system_name == "Selkov glycolysis model":
            self.num_states = 2
            self.test_ode = lambda t, x: odes.selkov(t, x)
            self.tf = 30.0
            self.x0 = [0.7, 1.25]
            self.noise = 1e-2
        elif system_name == "LotkaVolterra":
            self.test_ode = lambda t, x: odes.lotka_volterra(t,x)
            self.num_states = 2
            self.tf = 30.0
            self.x0 = [2.0, 1.0]
            self.noise = 1e-2

        data = torch.load(data_path)
        system_choice = data["name"]
        d = data["d"]
        self.num_data = data["num_data"]
        self.t = data["t"]
        self.H= data["G"]
        self.y_data = data["y_data"]  # 1 X seq_len X N
        self.true_data = data["data_true"]
        var = data["var"]
        noise_percent = data["noise_percent"]
        self.x0 = data["x0"]

        degree = data["degree"]
        Q = data["pnoise_cov"]
        self.n_obs = self.H.shape[0]
        self.device = "cpu"
        self.dtype = torch.float64
        self.Q = torch.diag(torch.tensor(Q, dtype=self.dtype))
        self.R = torch.diag(torch.tensor(var, dtype=self.dtype))

    def forward(self):
        def f_lv(x, t=0.0):
            X = self.test_ode(t, x)
            return np.array(X, dtype=np.float64)
            # 观测：线性取样 + 噪声（可替换为你的 h）
        def h_lv(x, dt=None):
            y = self.H @ x
            return y.detach().cpu().numpy().astype(np.float64)

        # 构造滤波器：连续微分方程 + RK4
        gmukf = RobustGMUKF_ODE(
            n_states=self.num_states,
            n_obs=self.n_obs,
            f=f_lv,  # 连续系统：dx/dt = f(x,t)
            h=h_lv,
            Q=self.Q,
            R=self.R,
            f_mode='dx',  # 'dx' 表示 f 返回 dx/dt
            integrator='rk4',
            kappa=0,
            alpha=1e-3,
            beta=2,
            huber_c=1.345,
            ps_window=1,
            ps_enable=True,
            ps_z_norm=True,
            irls_iters=2,
            device=self.device,
            dtype=self.dtype
        )
        Y = self.y_data
        batch = Y.shape[0]
        dt = self.t[:,1]-self.t[:,0]
        dt = np.array(dt, dtype=np.float64)

        X0 = torch.as_tensor(self.x0, dtype=torch.float64).unsqueeze(0)
        P0 = 1e-3 * torch.eye(self.num_states, dtype=self.dtype).unsqueeze(0).repeat(batch, 1, 1)

        X_est, P_est = gmukf.run(X0, P0, Y, dt=dt)

        loss = rmse2(self.true_data[0, 1:, :], X_est[0, 1:, :])
        self.loss = 10 * torch.log10(loss)

        loss2 = rmse(self.true_data[:, 1:, :], X_est[:, 1:, :])
        self.loss2 = 10 * torch.log10(loss2)

        P_est = P_est.squeeze(0).diagonal(dim1=-2, dim2=-1)

        return X_est, P_est, self.loss, self.loss2





