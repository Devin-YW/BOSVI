import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Optional
import numpy as np


# ========== 残差块 ==========
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


# ========== CNN-LSTM 漂移和扩散网络 ==========
class CNNLSTMSharedDriftDiffusion(nn.Module):
    def __init__(self, input_dim, cnn_channels=32, lstm_hidden=64, num_res_blocks=2):
        super().__init__()
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels

        # 输入卷积层
        self.conv_in = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)

        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(cnn_channels) for _ in range(num_res_blocks)]
        )

        # LSTM层
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)

        # 输出头
        self.drift_head = nn.Linear(lstm_hidden, input_dim)
        self.diff_head = nn.Linear(lstm_hidden, input_dim)  # 扩散项输出

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        处理批次时间序列输入

        Args:
            z (Tensor): 状态序列 (batch_size, seq_len, input_dim)

        Returns:
            drift (Tensor): 漂移 (batch_size, seq_len, input_dim)
            diffusion_matrix (Tensor): 扩散矩阵 (batch_size, seq_len, input_dim, input_dim)
        """
        batch_size, seq_len, _ = z.shape

        # 转置为通道优先格式: (batch_size, input_dim, seq_len)
        z_seq = z.transpose(1, 2)

        # CNN处理
        x = F.relu(self.conv_in(z_seq))  # (batch_size, cnn_channels, seq_len)
        x = self.res_blocks(x)  # (batch_size, cnn_channels, seq_len)

        # 转回为序列格式: (batch_size, seq_len, cnn_channels)
        x = x.transpose(1, 2)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden)

        # 输出头部
        drift = self.drift_head(lstm_out)  # (batch_size, seq_len, input_dim)

        # 扩散系数输出 (确保正定)
        diff_coeff = F.softplus(self.diff_head(lstm_out))  # (batch_size, seq_len, input_dim)

        # 构造对角扩散矩阵 (batch_size, seq_len, input_dim, input_dim)
        eye = torch.eye(self.input_dim, device=diff_coeff.device)
        diffusion_matrix = diff_coeff.unsqueeze(-1) * eye.unsqueeze(0).unsqueeze(0)

        return drift, diffusion_matrix

    def drift(self, t: Tensor, z: Tensor) -> Tensor:
        """
        计算漂移函数 (SDELearner 接口要求)

        Args:
            t (Tensor): 时间戳 (batch_size,)
            z (Tensor): 状态 (batch_size, input_dim)

        Returns:
            Tensor: 漂移 (batch_size, input_dim)
        """
        # 添加序列维度: (batch_size, 1, input_dim)
        z_expanded = z.unsqueeze(1)

        # 通过网络
        drift, _ = self.forward(z_expanded)

        # 移除序列维度: (batch_size, 1, input_dim) -> (batch_size, input_dim)
        return drift.squeeze(1)

    def diffusion(self) -> Tensor:
        """返回扩散矩阵的平均值 (SDELearner 接口要求)"""
        # 使用零状态作为参考
        dummy_z = torch.zeros(1, 1, self.input_dim, device=next(self.parameters()).device)
        dummy_z = dummy_z.to(next(self.parameters()).device)

        _, diffusion_matrix = self.forward(dummy_z)

        # 返回平均扩散矩阵 (input_dim, input_dim)
        return diffusion_matrix.mean(dim=0).mean(dim=0).squeeze()

    def resample_weights(self):
        """为了兼容性保留的空方法"""
        pass

    def kl_divergence(self) -> Tensor:
        """返回KL散度（这里为0，因为没有概率权重）"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ========== SDE 学习器 ==========
class NeuralSDE_CNNLSTM(nn.Module):
    def __init__(
            self,
            d: int,
            t_span: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
            n_reparam_samples: int = 8,
            G: torch.Tensor = None,
            measurement_noise: torch.Tensor = None,
            cnn_channels: int = 32,
            lstm_hidden: int = 64,
            num_res_blocks: int = 2,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.d = d
        self.n_reparam_samples = n_reparam_samples
        self.device = device

        # 处理 t_span 输入
        if isinstance(t_span[0], torch.Tensor):
            t_low = t_span[0].item()
            t_high = t_span[1].item()
        else:
            t_low, t_high = t_span
        self.t_span = (t_low, t_high)

        # 默认观测矩阵和噪声
        if G is None:
            G = torch.eye(d, device=device)
        if measurement_noise is None:
            measurement_noise = torch.ones(d, device=device) * 1e-2

        # 创建漂移+扩散模型
        self.sde_prior = CNNLSTMSharedDriftDiffusion(
            input_dim=d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            num_res_blocks=num_res_blocks
        ).to(device)

        # 保存参数
        self.G = G
        self.measurement_noise = measurement_noise

        # 用于高斯过程近似的参数
        n_points = 100  # 时间点数量
        self.tau = torch.linspace(t_low, t_high, n_points, device=device)
        self.mean = torch.zeros(n_points, d, device=device)

        # 创建协方差矩阵 (n_points, n_points, d, d)
        eye_time = torch.eye(n_points, device=device)  # (n_points, n_points)
        eye_state = torch.eye(d, device=device)  # (d, d)

        # 使用 einsum 正确组合维度
        # 结果形状: (n_points, n_points, d, d)
        self.cov = torch.einsum('ij,kl->ijkl', eye_time, eye_state) * 0.1

    def forward(self, t: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """计算漂移和扩散"""
        # 添加批次维度
        if z.dim() == 2:
            z = z.unsqueeze(0)

        # 通过网络
        drift, diffusion = self.sde_prior(z)

        # 如果输入没有批次维度，则移除
        if z.dim() == 2:
            drift = drift.squeeze(0)
            diffusion = diffusion.squeeze(0)

        return drift, diffusion

    def elbo(
            self,
            t_batch: Tensor,  # (batch_size, seq_len)
            y_batch: Tensor,  # (batch_size, seq_len, d)
            beta: float = 1.0,
            print_loss: bool = False
    ) -> Tensor:
        """
        计算证据下界 (ELBO)

        Args:
            t_batch: 时间戳 (batch_size, seq_len)
            y_batch: 观测值 (batch_size, seq_len, d)
            beta: KL散度权重
            print_loss: 是否打印损失值

        Returns:
            ELBO值
        """
        batch_size, seq_len, d = y_batch.shape
        N = batch_size * seq_len  # 总数据点数

        # 1. 状态估计 - 高斯过程近似
        # 这里简化处理，实际应用中应使用更复杂的近似
        x_mean, x_var = self._approximate_state(t_batch, y_batch)

        # 2. 重参数化采样
        epsilon = torch.randn(batch_size, seq_len, d, dtype=y_batch.dtype, device=y_batch.device)
        x_sample = x_mean + epsilon * torch.sqrt(x_var)

        # 3. 计算漂移和扩散
        drift, diffusion = self.forward(t_batch, x_sample)  # (batch_size, seq_len, d)

        # 4. 对数似然 (观测模型)
        # 简化观测模型: y = x + noise
        log_like = self._log_likelihood(y_batch, x_sample)

        # 5. SDE残差
        # 计算状态变化
        dx = x_sample[:, 1:] - x_sample[:, :-1]
        dt = t_batch[:, 1:] - t_batch[:, :-1]
        drift_avg = 0.5 * (drift[:, :-1] + drift[:, 1:])

        # 残差: dx/dt - f(x)
        residual = (dx / dt.unsqueeze(-1)) - drift_avg
        residual_loss = torch.mean(residual ** 2)

        # 6. KL散度 (简化)
        # 在实际应用中应使用更准确的KL计算
        kl_divergence = torch.tensor(0.0, device=y_batch.device)

        # 7. 组合ELBO
        elbo_val = log_like - beta * (0.5 * residual_loss + kl_divergence / N)

        if print_loss:
            print(f"ELBO: {elbo_val.item():.4f}, "
                  f"LogLike: {log_like.item():.4f}, "
                  f"Residual: {residual_loss.item():.6f}, "
                  f"KL: {kl_divergence.item():.6f}")

        return elbo_val

    def _log_likelihood(self, y: Tensor, x: Tensor) -> Tensor:
        """计算对数似然"""
        # 简化: 高斯观测噪声
        residuals = y - x
        log_like = -0.5 * torch.sum(residuals ** 2 / self.measurement_noise ** 2)
        log_like -= 0.5 * torch.sum(torch.log(2 * np.pi * self.measurement_noise ** 2))
        return log_like / y.shape[0]  # 按批次大小归一化

    def _approximate_state(self, t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """高斯过程状态近似 (简化版)"""
        batch_size, seq_len, d = y.shape

        # 简化处理: 使用观测值的加权平均作为状态估计
        x_mean = torch.zeros_like(y)
        x_var = torch.ones_like(y) * 0.1

        # 使用观测值作为初始估计
        x_mean = y.clone()

        # 简单平滑
        if seq_len > 1:
            # 简单移动平均
            x_mean[:, 1:-1] = 0.25 * y[:, :-2] + 0.5 * y[:, 1:-1] + 0.25 * y[:, 2:]

        return x_mean, x_var

    def predict(self, t: Tensor, initial_state: Tensor, steps: int) -> Tensor:
        """
        预测未来状态

        Args:
            t: 当前时间 (batch_size,)
            initial_state: 初始状态 (batch_size, d)
            steps: 预测步数

        Returns:
            预测状态序列 (batch_size, steps, d)
        """
        batch_size = initial_state.shape[0]
        states = torch.zeros(batch_size, steps, self.d, device=self.device)
        current_state = initial_state.clone()

        # 时间步长 (假设恒定)
        dt = (self.t_span[1] - self.t_span[0]) / 100

        for i in range(steps):
            # 计算漂移和扩散
            drift, diffusion = self.sde_prior.drift(t, current_state), self.sde_prior.diffusion()

            # Euler-Maruyama方法
            noise = torch.randn_like(current_state) * torch.sqrt(dt)
            current_state = current_state + drift * dt + diffusion.matmul(noise.unsqueeze(-1)).squeeze()

            # 存储状态
            states[:, i] = current_state

            # 更新时间
            t += dt

        return states


# ========== 数据预处理工具函数 ==========
def create_batch_sequences(t: np.ndarray, y: np.ndarray, seq_len: int, batch_size: int):
    """
    从完整序列创建批次数据

    Args:
        t: 时间戳数组 (N,)
        y: 观测值数组 (N, d)
        seq_len: 序列长度
        batch_size: 批次大小

    Returns:
        t_batch: (batch_size, seq_len)
        y_batch: (batch_size, seq_len, d)
    """
    N, d = y.shape
    num_batches = (N - seq_len) // batch_size

    # 创建索引
    indices = np.random.choice(N - seq_len, num_batches * batch_size, replace=False)

    # 创建批次数据
    t_batch = np.zeros((batch_size * num_batches, seq_len))
    y_batch = np.zeros((batch_size * num_batches, seq_len, d))

    for i, start in enumerate(indices):
        batch_idx = i // batch_size
        item_idx = i % batch_size

        end = start + seq_len
        t_batch[i] = t[start:end]
        y_batch[i] = y[start:end]

    return (
        torch.tensor(t_batch, dtype=torch.float32),
        torch.tensor(y_batch, dtype=torch.float32)
    )


# ========== 示例训练代码 ==========
def train_example():
    # 1. 创建模拟数据
    d = 3  # 状态维度
    N = 1000  # 数据点数量
    t = np.linspace(0, 10, N)
    y = np.zeros((N, d))

    # 创建螺旋轨迹
    y[:, 0] = np.sin(t)
    y[:, 1] = np.cos(t)
    y[:, 2] = t / 10.0

    # 2. 创建批次数据
    seq_len = 50  # 序列长度
    batch_size = 16  # 批次大小
    t_batch, y_batch = create_batch_sequences(t, y, seq_len, batch_size)

    # 3. 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralSDE_CNNLSTM(
        d=d,
        t_span=(t.min(), t.max()),
        n_reparam_samples=8,
        device=device
    ).to(device)

    # 4. 训练参数
    num_epochs = 100
    warmup_iters = 20
    learning_rate = 1e-3
    print_interval = 10

    # 5. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 6. 训练循环
    for epoch in range(num_epochs):
        # 移动到设备
        t_batch_device = t_batch.to(device)
        y_batch_device = y_batch.to(device)

        # 计算beta
        beta = min(1.0, (epoch + 1) / warmup_iters)

        # 前向传播
        optimizer.zero_grad()
        loss = -model.elbo(t_batch_device, y_batch_device, beta)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印训练进度
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Beta: {beta:.2f}")

    # 7. 保存模型
    torch.save(model.state_dict(), "cnn_lstm_sde_model.pth")
    print("训练完成，模型已保存!")

    # 8. 示例预测
    initial_t = torch.tensor([0.0], device=device)
    initial_state = torch.tensor([[0.0, 1.0, 0.0]], device=device)  # 初始状态

    # 预测未来100步
    with torch.no_grad():
        predictions = model.predict(initial_t, initial_state, steps=100)

    print(f"预测状态形状: {predictions.shape}")


if __name__ == "__main__":
    train_example()