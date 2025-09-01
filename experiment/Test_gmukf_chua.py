from SviKalmanNet.GM_UKF import RobustGMUKF_ODE
import numpy as np
from NRMSE import rmse, rmse2
import torch
class Tester_GM_UKF:
    def __init__(self,model_name, model_path, data_path):
        if model_name == "Chua_circuit":
            model = torch.load(model_path)
            model_name = model["name"]
            self.num_states = model["d"]
            tmin = model["t0"]
            tmax = model["tf"]
            self.x0 = model["x0"]
            self.var = model["var"]
            self.Q = model["pnoise_cov"]
            self.H = model["G"]
            degree = model["degree"]

        self.t = torch.load(data_path + "time.pt")
        self.y_data = torch.load(data_path+"obs.pt")
        self.true_data = torch.load(data_path + "true_state.pt")
        self.n_obs = self.H.shape[0]
        self.device = "cpu"
        self.dtype = torch.float64
        self.Q = torch.diag(torch.tensor(self.Q, dtype=self.dtype))
        self.R = torch.diag(torch.tensor(self.var, dtype=self.dtype))

    def chua_circuit(self, t, x):
        R = 1600
        C1 = 10
        C2 = 100
        L = 20e-3
        u1, u2 = x[..., 0], x[..., 1]
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

    def forward(self):
        def f_lv(x, t=0.0):
            X = self.chua_circuit(t, x)
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
            ps_window=4,
            ps_enable=True,
            ps_z_norm=True,
            irls_iters=2,
            device=self.device,
            dtype=self.dtype
        )
        Y = self.y_data
        batch = Y.shape[0]
        dt = 0.02
        dt = np.array(dt, dtype=np.float64)

        X0 = torch.as_tensor(self.x0.squeeze(), dtype=torch.float64).unsqueeze(0)
        P0 = 1e-3 * torch.eye(self.num_states, dtype=self.dtype).unsqueeze(0).repeat(batch, 1, 1)

        X_est, P_est = gmukf.run(X0, P0, Y, dt=dt)

        loss = rmse2(self.true_data[0, 1:, :], X_est[0, 1:, :])
        self.loss = 10 * torch.log10(loss)
        loss2 = rmse(self.true_data[:, 1:, :], X_est[:, 1:, :])
        self.loss2 = 10 * torch.log10(loss2)

        return X_est.detach().cpu().numpy(), P_est.detach().cpu().numpy(), self.loss.detach().cpu().numpy(),self.loss2.detach().cpu().numpy()