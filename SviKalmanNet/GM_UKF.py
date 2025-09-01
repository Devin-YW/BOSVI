import torch
import math
import numpy as np
from collections import deque
import torch.nn as nn

# =========================
# Robust工具：Huber / MAD / PS
# =========================

def huber_weight(r, c):
    """
    r: 标准化残差（tensor）
    c: Huber阈值
    返回：与r同形状的权重 in (0,1]
    """
    r = r.to(torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64, device=r.device)
    abs_r = torch.abs(r)
    w = torch.ones_like(r, dtype=torch.float64)
    mask = abs_r > c
    w[mask] = c / (abs_r[mask] + 1e-12)
    return w

def robust_loc_scale(Z):
    """
    Z: (num_samples, dim) 窗口样本
    返回：列方向的鲁棒位置(中位数)与尺度(MAD*1.4826)
    """
    Z = Z.to(torch.float64)
    mu = torch.median(Z, dim=0).values
    mad = torch.median(torch.abs(Z - mu), dim=0).values
    sigma = 1.4826 * mad
    sigma = torch.clamp(sigma, min=1e-8)
    return mu, sigma

def projection_stat_weight(z_t, Zwin, c_ps=2.5):
    """
    简化版 PS：用鲁棒位置/尺度把 z_t 标准化，得标量异常度 -> Huber权重
    z_t: (dim,)
    Zwin: (w, dim)
    c_ps: 阈值
    """
    mu, sigma = robust_loc_scale(Zwin)
    z_std = (z_t - mu) / sigma
    r = torch.linalg.norm(z_std)  # 标量
    w = huber_weight(r, c_ps)     # 标量权重
    return w

# ======================================
#   可处理微分系统（ODE）的 Robust GM-UKF
# ======================================

class RobustGMUKF_ODE:
    def __init__(self,
                 n_states, n_obs,
                 f,               # 如果 f_mode='dx'，f(x, t) -> dx/dt；若 f_mode='step'，f(x, dt) -> x_next
                 h,               # h(x, dt) -> y
                 Q, R,
                 f_mode='dx',     # 'dx' (默认, 连续微分方程) 或 'step' (已离散一步)
                 integrator='rk4',
                 kappa=0, alpha=1e-3, beta=2,
                 huber_c=1.345,   # Huber阈值
                 ps_window=30,    # PS滑动窗口长度
                 ps_enable=True,
                 ps_z_norm=True,
                 irls_iters=2,    # IRLS内部迭代 1~3 通常足够
                 device="cpu",
                 dtype=torch.float64):
        self.n_states = n_states
        self.n_obs = n_obs
        self.f = f
        self.h = h
        self.f_mode = f_mode
        self.integrator = integrator
        self.device = device
        self.dtype = dtype

        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self.Q = self._to_cov(Q, n_states)
        self.R = self._to_cov(R, n_obs)

        self.huber_c = float(huber_c)
        self.ps_enable = bool(ps_enable)
        self.ps_window = int(ps_window)
        self.ps_z_norm = bool(ps_z_norm)
        self.irls_iters = int(irls_iters)

        self.Z_buffer = deque(maxlen=self.ps_window)

    # ---------- 基础工具 ----------
    def _to_cov(self, x, dim):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype)
        if x.dim() == 0:
            x = torch.eye(dim, dtype=self.dtype) * x
        elif x.dim() == 1 and x.shape[0] == 1:
            x = torch.eye(dim, dtype=self.dtype) * x[0]
        return x.to(self.device, dtype=self.dtype)

    def _sigma_points(self, x, P):
        n = x.shape[0]
        lam = self.alpha**2 * (n + self.kappa) - n
        c = n + lam
        # 这里把 (P*c) 做cholesky，省一次乘 sqrt(c)
        chol = torch.linalg.cholesky((P * c).to(self.dtype))
        pts = [x.unsqueeze(0)]
        for i in range(n):
            ei = chol[:, i]
            pts.append((x + ei).unsqueeze(0))
            pts.append((x - ei).unsqueeze(0))
        X = torch.cat(pts, dim=0)  # [2n+1, n]

        Wm = torch.full((2*n+1,), 1.0/(2.0*c), dtype=self.dtype, device=self.device)
        Wc = torch.full((2*n+1,), 1.0/(2.0*c), dtype=self.dtype, device=self.device)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1 - self.alpha**2 + self.beta)
        return X, Wm, Wc

    def _ut(self, Xsigma, Wm, Wc, noise=None):
        mu = torch.sum(Wm.view(-1,1) * Xsigma, dim=0)
        diff = Xsigma - mu
        P = torch.zeros(mu.shape[0], mu.shape[0], dtype=self.dtype, device=self.device)
        for i in range(Xsigma.shape[0]):
            P += Wc[i] * (diff[i].unsqueeze(1) @ diff[i].unsqueeze(0))
        if noise is not None:
            P = P + noise
        return mu, P

    # ---------- ODE 积分器 ----------
    def _rk4_step(self, x, dt):
        """
        x: torch.tensor(n_states,)
        f: dx/dt = f(x, t) —— 这里不显含 t 的就忽略 t
        """
        x_np = x.detach().cpu().numpy()
        k1 = self.f(x_np, 0.0)
        k2 = self.f(x_np + 0.5*dt*np.asarray(k1), 0.0)
        k3 = self.f(x_np + 0.5*dt*np.asarray(k2), 0.0)
        k4 = self.f(x_np + dt*np.asarray(k3), 0.0)
        x_next = x_np + (dt/6.0)*(np.asarray(k1) + 2*np.asarray(k2) + 2*np.asarray(k3) + np.asarray(k4))
        return torch.tensor(x_next, dtype=self.dtype, device=self.device)

    def _f_step(self, x, dt):
        """
        返回一步后的状态：
        - 若 f_mode='dx'，用 RK4 对 ODE 积分一步；
        - 若 f_mode='step'，假设 f(x, dt)->x_next。
        """
        if self.f_mode == 'dx':
            return self._rk4_step(x, dt)
        else:
            x_np = x.detach().cpu().numpy()
            x_next = self.f(x_np, dt)
            return torch.tensor(x_next, dtype=self.dtype, device=self.device)

    # ---------- 主滤波 ----------
    def run(self, X_init, P_init, Y, dt=0.02):
        """
        X_init: (batch, n_states)
        P_init: (batch, n_states, n_states)
        Y:      (batch, seq_len, n_obs)
        返回: X_est, P_est
        """
        batch, seq_len, _ = Y.shape
        X_init = X_init.to(self.device, dtype=self.dtype)
        if P_init is None:
            P_all = torch.eye(self.n_states, dtype=self.dtype, device=self.device)[None].repeat(batch,1,1)
        else:
            P_all = P_init.to(self.device, dtype=self.dtype)
        Y = Y.to(self.device, dtype=self.dtype)

        X_out, P_out = [], []

        for b in range(batch):
            x = X_init[b].clone()
            P = P_all[b].clone()
            self.Z_buffer.clear()

            xs, Ps = [], []

            for t in range(seq_len):
                # 1) sigma 点 & 传播
                Xsigma, Wm, Wc = self._sigma_points(x, P)

                # 状态传播：ODE 一步/离散一步
                fx = []
                for i in range(Xsigma.shape[0]):
                    fi = self._f_step(Xsigma[i], dt)
                    fx.append(fi)
                fx = torch.stack(fx, dim=0)  # [2n+1, n_states]
                x_pred, P_pred = self._ut(fx, Wm, Wc, noise=self.Q)

                # 量测传播
                hx = []
                for i in range(fx.shape[0]):
                    yi = self.h(fx[i].detach().cpu().numpy(), dt)
                    yi = torch.tensor(yi, dtype=self.dtype, device=self.device)
                    hx.append(yi)
                hx = torch.stack(hx, dim=0)  # [2n+1, n_obs]
                y_pred, Pyy = self._ut(hx, Wm, Wc, noise=self.R)

                # 交叉协方差
                dx = fx - x_pred
                dy = hx - y_pred
                Pxy = torch.zeros(self.n_states, self.n_obs, dtype=self.dtype, device=self.device)
                for i in range(fx.shape[0]):
                    Pxy += Wc[i] * (dx[i].unsqueeze(1) @ dy[i].unsqueeze(0))

                # 创新
                y_obs = Y[b, t]
                innov = y_obs - y_pred  # (n_obs,)

                # 2) 构造 PS 的 z_t 并入窗
                z_t = torch.cat([x_pred, innov], dim=0)  # (n_states + n_obs,)
                if self.ps_z_norm:
                    scale = torch.clamp(torch.abs(z_t), min=1e-3)
                    z_tn = z_t / scale
                else:
                    z_tn = z_t

                if self.ps_enable:
                    self.Z_buffer.append(z_tn.detach())
                    Zwin = torch.stack(list(self.Z_buffer), dim=0)  # (w, n_states+n_obs)

                # 3) IRLS 鲁棒更新
                x_cur, P_cur = x_pred.clone(), P_pred.clone()

                for _ in range(self.irls_iters):
                    # PS 标量权重
                    if self.ps_enable and len(self.Z_buffer) >= 5:  # 窗口太短时不给PS太大权重
                        w_ps = projection_stat_weight(z_tn, Zwin, c_ps=2.5)
                        w_ps = torch.clamp(w_ps, min=1e-3)  # 数值保护
                        w_ps = w_ps.to(self.device, dtype=self.dtype)
                    else:
                        w_ps = torch.as_tensor(1.0, dtype=self.dtype, device=self.device)

                    # 分量级 Huber 权重（基于 diag(Pyy) 的标准化）
                    diag = torch.clamp(torch.diag(Pyy), min=1e-10)
                    r_std = innov / torch.sqrt(diag)
                    w_comp = huber_weight(r_std, self.huber_c)  # (n_obs,)

                    # 组合为对角权重矩阵
                    w_vec = w_ps * w_comp
                    W_meas = torch.diag(w_vec)

                    # 因 W_meas 对角，平方根简单取 sqrt(diag)
                    Wsqrt = torch.diag(torch.sqrt(torch.clamp(w_vec, min=1e-8)))

                    # 加权互协方与测量协方
                    Pxy_rob = Pxy @ Wsqrt
                    Pyy_rob = (Wsqrt.T @ Pyy @ Wsqrt) + self.R

                    # Kalman 增益（鲁棒）
                    K = Pxy_rob @ torch.linalg.inv(Pyy_rob) @ Wsqrt.T

                    # 更新
                    x_cur = x_pred + K @ innov
                    # 协方差的“Joseph”形式更稳健，这里用常见等价写法
                    P_cur = P_pred - K @ Pyy @ K.T

                    # （可选）刷新统计再做下一次 IRLS迭代
                    # 简化：创新、y_pred 不重算；通常 1-2 次就很稳

                # 本时刻完成
                x, P = x_cur, P_cur
                xs.append(x)
                Ps.append(P)

            X_out.append(torch.stack(xs, dim=0))
            P_out.append(torch.stack(Ps, dim=0))

        return torch.stack(X_out, dim=0), torch.stack(P_out, dim=0)

# =========================
# 使用示例（微分系统：Lotka–Volterra）
# =========================
if __name__ == "__main__":
    """
    示例：Lotka–Volterra 捕食-被捕食模型（连续微分），
    我们用 GM-UKF（ODE+RK4）来做状态估计。
    你可以把 f_dyn/h 换成你的电力系统微分方程与 PMU 量测模型即可。
    """
    # dx/dt = [ a*x - b*x*y,
    #           -c*y + d*x*y ]
    a, b, c, d = 1.1, 0.4, 0.4, 0.1

    def f_lv(x, t=0.0):
        x1, x2 = x[0], x[1]
        dx1 = a*x1 - b*x1*x2
        dx2 = -c*x2 + d*x1*x2
        return np.array([dx1, dx2], dtype=np.float64)

    # 观测：线性取样 + 噪声（可替换为你的 h）
    H = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    def h_lv(x, dt=None):
        return (H @ x).astype(np.float64)

    device = "cpu"
    dtype = torch.float64

    n_states, n_obs = 2, 2
    Q = torch.diag(torch.tensor([1e-4, 1e-4], dtype=dtype))
    R = torch.diag(torch.tensor([4e-3, 4e-3], dtype=dtype))

    # 构造滤波器：连续微分方程 + RK4
    gmukf = RobustGMUKF_ODE(
        n_states=n_states,
        n_obs=n_obs,
        f=f_lv,             # 连续系统：dx/dt = f(x,t)
        h=h_lv,
        Q=Q,
        R=R,
        f_mode='dx',        # 'dx' 表示 f 返回 dx/dt
        integrator='rk4',
        kappa=0,
        alpha=1e-3,
        beta=2,
        huber_c=1.345,
        ps_window=40,
        ps_enable=True,
        ps_z_norm=True,
        irls_iters=2,
        device=device,
        dtype=dtype
    )

    # 造数据
    rng = np.random.default_rng(0)
    batch, seq_len, dt = 3, 80, 0.02
    X_true = torch.zeros(batch, seq_len, n_states, dtype=dtype)
    Y = torch.zeros(batch, seq_len, n_obs, dtype=dtype)

    for b in range(batch):
        x = np.array([1.5 + 0.2*b, 1.0], dtype=np.float64)
        for t in range(seq_len):
            # 真值前向RK4积分
            # （与滤波器内部一致以便演示；真实场景可用更高精度积分）
            def rk4(x, dt):
                k1 = f_lv(x)
                k2 = f_lv(x + 0.5*dt*k1)
                k3 = f_lv(x + 0.5*dt*k2)
                k4 = f_lv(x + dt*k3)
                return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            x = rk4(x, dt) + rng.normal(0, [3e-3, 3e-3])  # 过程噪声
            y = h_lv(x) + rng.normal(0, [0.06, 0.06])     # 观测噪声

            # 注入离群点
            if t in [20, 45] and b == 0:
                y += np.array([0.5, -0.6])

            X_true[b, t] = torch.tensor(x, dtype=dtype)
            Y[b, t] = torch.tensor(y, dtype=dtype)

    # 初始
    X0 = torch.tensor([[1.4, 1.1]]*batch, dtype=dtype)
    P0 = torch.eye(n_states, dtype=dtype).unsqueeze(0).repeat(batch,1,1)

    X_est, P_est = gmukf.run(X0, P0, Y, dt=dt)

    mse = nn.MSELoss()(X_est, X_true).item()
    print(f"[Robust GM-UKF (ODE+RK4)] MSE = {mse:.6f}")
