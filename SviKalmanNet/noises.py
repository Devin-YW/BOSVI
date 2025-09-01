import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable

class Noise:
    def __init__(self, data, noise_type, snr):
        self.data = data
        self.data_size = data.size()
        self.noise_type = noise_type
        self.snr = snr

        # 计算信号功率
        self.signal_power = torch.mean(self.data ** 2, dim=0)
        self.noise_power = self.signal_power / (10 ** (self.snr / 10))
        self.noise_std = self.noise_power.sqrt()

        var1, var2 = torch.var(self.data, dim=0)
        # 计算噪声功率
        self.noise_power1 = var1 / (10 ** (self.snr / 10))
        self.noise_std1 = self.noise_power1.sqrt()  # 噪声标准差
        self.noise_power2 = var2 / (10 ** (self.snr / 10))
        self.noise_std2 = self.noise_power2.sqrt()  # 噪声标准差
        self.noise_var = torch.stack((self.noise_power1, self.noise_power2), dim=0)
        #


    def add_to_noise(self):
        # SNR
        # self.data
        # data_range = (self.data.max(0).values - (self.data).min(0).values) / 2
        # mea_noise_percent = 0.01
        # var = (mea_noise_percent * data_range) ** 2
        if self.noise_type == "Gaussian":

            gaussian_noise1 = torch.normal(0, self.noise_std1, size=self.data[:, 0].shape)
            noise_data1 = gaussian_noise1 + self.data[:,0]
            gaussian_noise2 = torch.normal(0, self.noise_std2, size=self.data[:, 1].shape)
            noise_data2 = gaussian_noise2 + self.data[:, 1]

            noise_data = torch.cat((noise_data1.unsqueeze(1), noise_data2.unsqueeze(1)), dim=1)
            # 绘图
            # plt.hist(gaussian_noise.numpy(), bins=30, density=True)
            # plt.title('Gaussian Distribution (Normal Distribution)')
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.show()
        elif self.noise_type == "Rayleigh":
            noise_list = []
            for i in range(self.data.shape[1]):  # 遍历每个维度
                # 生成独立的正态分布分量
                x = torch.normal(0, 1, size=self.data[:, i].shape)
                y = torch.normal(0, 1, size=self.data[:, i].shape)
                # 计算瑞利噪声
                dim_noise = torch.sqrt(x ** 2 + y ** 2)
                # 标准化并缩放至目标功率
                dim_noise = (dim_noise - dim_noise.mean()) / dim_noise.std() * self.noise_var[i].sqrt()
                noise_list.append(dim_noise.unsqueeze(1))

            noise = torch.cat(noise_list, dim=1)
            noise_data = self.data + noise
            # # 参数
            # mean1 = 0
            # mean2 = 0
            # std1 = 1
            # std2 = 1
            # # 生成两个独立的正态分布数据
            # x = torch.normal(mean1, std1, size=self.data_size)
            # y = torch.normal(mean2, std2, size=self.data_size)
            #
            # # 计算模长，即瑞利分布
            # rayleigh_noise = torch.sqrt(x ** 2 + y ** 2)
            # noise = (rayleigh_noise - rayleigh_noise.mean()) / rayleigh_noise.std() * self.noise_std
            # noise_data = self.data + noise
            # 绘图
            # for i in range(size[0]):  # 遍历每一行
            #     plt.scatter(torch.arange(size[1]).numpy(), rayleigh_noise[i].numpy(), label=f'Sample {i + 1}', s=10)
            #
            # plt.title('Rayleigh Distribution')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.show()

        elif self.noise_type == "Alpha-Stable":

            noise_list = []
            for i in range(self.data.shape[1]):
                # 独立生成各维度噪声
                stable_noise = levy_stable.rvs(alpha=1.5, beta=0.0, size=self.data[:, i].shape)
                stable_tensor = torch.tensor(stable_noise, dtype=torch.float32)
                # 分维度标准化
                stable_tensor = (stable_tensor - stable_tensor.mean()) / stable_tensor.std() * self.noise_var[i].sqrt()
                noise_list.append(stable_tensor.unsqueeze(1))

            noise = torch.cat(noise_list, dim=1)
            noise_data = self.data + noise

            # # 参数
            # alpha = 1.5  # 稳定指数，通常范围 [0, 2]
            # beta = 0  # 偏度参数，通常范围 [-1, 1]
            #
            # # 生成 alpha-stable 分布数据
            # stable_noise = levy_stable.rvs(alpha, beta, size=self.data_size)
            #
            # # 转换为 PyTorch 张量
            # stable_noise_tensor = torch.tensor(stable_noise, dtype=torch.float32)
            # noise = (stable_noise_tensor - stable_noise_tensor.mean()) / stable_noise_tensor.std() * self.noise_std
            # noise_data = self.data + noise
            # 绘图
            # plt.hist(stable_noise_tensor.numpy(), bins=30, density=True)
            # plt.plot(stable_noise_tensor.numpy())  # 直接绘制数据
            # plt.scatter(torch.arange(size).numpy(), stable_noise_tensor.numpy(), s=1)  # s=1 控制点的大小
            # plt.title('Alpha-Stable Distribution')
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.show()
        elif self.noise_type == "Uniform":
            noise_list = []
            for i in range(self.data.shape[1]):
                # 计算各维度范围参数
                a = -np.sqrt(3 * self.noise_var[i].item())  # 均匀分布方差公式： (b-a)^2/12
                b = np.sqrt(3 * self.noise_var[i].item())
                # 分维度生成
                dim_noise = torch.empty(self.data[:, i].shape).uniform_(a, b)
                noise_list.append(dim_noise.unsqueeze(1))

            noise = torch.cat(noise_list, dim=1)
            noise_data = self.data + noise
            # # 生成均匀分布噪声
            # a = float(-self.noise_std[1].item() * (2 ** 0.5))
            #
            # b = float(self.noise_std[1].item() * (2 ** 0.5))
            # # 生成均匀分布噪声
            # uniform_noise = torch.empty(self.data_size).uniform_(a, b)
            # noise_data = self.data + uniform_noise
            # 绘图
            # plt.hist(uniform_noise.numpy(), bins=30, density=True, alpha=0.6, color='g')
            # plt.title('Uniform Distribution Noise')
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.grid(True)
            # plt.show()
        elif self.noise_type == "Laplace":
            noise_list = []
            for i in range(self.data.shape[1]):
                # 根据目标方差计算尺度参数
                scale = torch.sqrt(self.noise_var[i] / 2)  # 拉普拉斯方差公式：2*scale^2
                # 分维度生成
                dim_noise = torch.distributions.Laplace(0, scale).sample(self.data[:, i].shape)
                noise_list.append(dim_noise.unsqueeze(1))
            noise = torch.cat(noise_list, dim=1)
            noise_data = self.data + noise

            # loc = torch.tensor(0.0)  # 中心位置
            # scale = torch.tensor(1)  # 尺度参数，控制噪声的幅度
            #
            # laplace_noise = torch.distributions.Laplace(loc, scale).sample(self.data_size)
            #
            # original_std_laplace = torch.sqrt(torch.tensor(2.0) * scale ** 2)  # sqrt(2) * scale
            #
            # zero_mean_noise_laplace = (laplace_noise - loc) / original_std_laplace
            #
            # scaled_noise_laplace = zero_mean_noise_laplace * self.noise_std
            #
            # noise_data = self.data + scaled_noise_laplace
        elif self.noise_type == "Bernoulli":
            noise_list = []
            for i in range(self.data.shape[1]):
                # 计算各维度伯努利参数
                p = 0.3  # 可扩展为向量化参数
                dim_noise = torch.bernoulli(torch.ones_like(self.data[:, i]) * p)
                # 标准化并缩放
                dim_noise = (dim_noise - p) / torch.sqrt(torch.tensor(p * (1 - p))) * self.noise_var[i].sqrt()
                noise_list.append(dim_noise.unsqueeze(1))

            noise = torch.cat(noise_list, dim=1)
            noise_data = self.data + noise
            # p = 0.3
            # bernoulli_noise = torch.distributions.Bernoulli(p).sample(self.data_size)  # 生成 0 或 1 的噪声
            #
            # original_std = torch.sqrt(torch.tensor(p * (1 - p)))  # sqrt(p(1-p))
            #
            # zero_mean_noise = (bernoulli_noise - p) / original_std
            #
            # scaled_noise = zero_mean_noise * self.noise_std
            #
            # noise_data = self.data + scaled_noise
        else:
            raise ValueError("Unsupported noise type!")

        return noise_data




