import torch
import torch.nn as nn


class my_lossFunc(nn.Module):
    def __init__(self, regularization_type="L2", lambda_reg=1e-4):
        """
        Args:
            regularization_type: 正则化类型 ("L1" 或 "L2")
            lambda_reg: 正则化强度
        """
        super(my_lossFunc, self).__init__()
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.robust_delta = 1.0

    def robust_loss(self, error):
        """
        使用 Huber 损失计算鲁棒性误差
        Args:
            error: 输入误差 (张量)
        Returns:
            loss: 鲁棒性损失
        """
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.robust_delta, device=error.device))
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + self.robust_delta * linear

    def forward(self, H, y, x_hat, x_prev, R_inv, P_inv, model_params=None):
        """
        Args:
            H: 观测矩阵 (B, B)
            y: 观测值 (A, B, C)
            x_hat: 当前估计状态 (A, B, C)
            x_prev: 上一状态 (A, B, C)
            R_inv: 观测噪声协方差的逆矩阵 (B, B)
            P_inv: 状态噪声协方差的逆矩阵 (B, B)
            model_params: 模型的参数列表，用于正则化 (torch.nn.Module.parameters())
        Returns:
            loss: 代价函数值 (标量)
        """
        A, B, C = y.shape  # 样本数量、状态维度、时间步数
        total_loss = 0.0

        for a in range(A):
            for k in range(C):
                # 当前时间步的观测误差
                observation_error = y[a, :, k] - torch.matmul(H, x_hat[a, :, k])  #
                # term1 = observation_error @ R_inv @ observation_error
                term1 = observation_error @ torch.inverse(R_inv) @ observation_error.t()

                # 当前时间步的状态误差
                state_error = x_hat[a, :, k] - x_prev[a, :, k]  #
                # term2 = state_error @ P_inv[a, k] @ state_error
                term2 = state_error @ torch.inverse(P_inv[a, k]) @ state_error.t()
                if k == 0:
                    term2 = 0.0
                    # state_loss = torch.sum(self.robust_loss(term2))
                # obs_lss = torch.sum(self.robust_loss(term1))

                total_loss += (term1 + term2)

        total_loss /= A * C

        if model_params is not None:
            reg_loss = 0.95
            for param in model_params:
                if self.regularization_type == "L2":
                    reg_loss += torch.sum(param ** 2)  # L2 范数平方
                elif self.regularization_type == "L1":
                    reg_loss += torch.sum(torch.abs(param))  # L1 范数
            total_loss += self.lambda_reg * reg_loss

        return total_loss


