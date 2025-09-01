import torch

def low_rank_obs_matrix(x_dim, r, device='cpu'):
    """
    生成低秩观测矩阵
    :param r: 目标秩
    :param device: 张量设备 (cpu/cuda)
    :return: 观测矩阵 G ∈ (r×4)
    """
    # 生成随机基向量并正交化
    U = torch.randn(x_dim, r, device=device)  # 标准正态分布采样
    Q, _ = torch.linalg.qr(U, mode='reduced')  # QR分解

    return Q.T