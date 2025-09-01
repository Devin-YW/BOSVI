import torch.multiprocessing.reductions
from scipy.io import loadmat
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
degree = 5
fs = 10   # 10 / 20
# load .mat file
data = loadmat(f'Chua_data_{fs}ms.mat')
print(data.keys())
t = data['t_span'].flatten()
print(t.shape)
true_data = data['true_data']
print(true_data.shape)
noise_data = data['noise_data']
print(noise_data.shape)

true_data = true_data.T   # (n, 2)
noise_data = noise_data.T

scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
true_data_normalized = scaler.fit_transform(true_data)
noise_data_normalized = scaler.fit_transform(noise_data)
noise = noise_data_normalized - true_data_normalized

var = np.cov(noise, rowvar=False)


def split_dataset(dataset, time, train_ratio=0.6, val_ratio=0.2):
    total_samples = len(dataset)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]

    train_t = time[:train_end]
    valid_t = time[train_end:val_end]
    test_t = time[val_end:]

    return train_dataset, val_dataset, test_dataset, train_t, valid_t, test_t

train_data, valid_data, test_data, _, _, _ = split_dataset(noise_data_normalized, t)
train_true, valid_true, test_true,  train_t, valid_t, test_t = split_dataset(true_data_normalized, t)

print("归一化后的数据形状：", true_data_normalized.shape)
print("归一化后的数据形状：", noise_data_normalized.shape)
print("训练集数据形状：", train_data.shape)
print("测试集数据形状：", test_data.shape)
print(valid_data.shape)
print(len(train_data) == len(train_true))
print(len(test_data) == len(test_true))
print(len(train_data) == len(train_t))

'''
import matplotlib.pyplot as plt

# 原始数据
plt.figure(figsize=(10, 6))
plt.plot(t, noise_data_normalized[:, 0], label='y1 (原始)')
plt.plot(t, noise_data_normalized[:, 1], label='y2 (原始)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Original Data')
plt.legend()
plt.grid(True)
plt.show()

# 归一化数据
plt.figure(figsize=(10, 6))
plt.plot(t, true_data_normalized[:, 0], label='y1 (归一化)')
plt.plot(t, true_data_normalized[:, 1], label='y2 (归一化)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Voltage')
plt.title('Normalized Data')
plt.legend()
plt.grid(True)
plt.show()
'''
# 转换为 tensor 保存
train_t = torch.tensor(train_t,  dtype=torch.float64).unsqueeze(0)
train_data = torch.tensor(train_data, dtype=torch.float64).unsqueeze(0)
train_true = torch.tensor(train_true, dtype=torch.float64).unsqueeze(0)
length = train_t.shape[1]

# print(train_t2.shape)
# print(train_data2.shape)
# print(train_true2.shape)
valid_t = torch.tensor(valid_t,  dtype=torch.float64).unsqueeze(0)
valid_data = torch.tensor(valid_data, dtype=torch.float64).unsqueeze(0)
valid_true = torch.tensor(valid_true, dtype=torch.float64).unsqueeze(0)

test_t = torch.tensor(test_t,  dtype=torch.float64).unsqueeze(0)
test_data = torch.tensor(test_data, dtype=torch.float64).unsqueeze(0)
test_true = torch.tensor(test_true, dtype=torch.float64).unsqueeze(0)

dt = 0.02
t0 = float(train_t.min())
tf = float(train_t.max())
# x0 = torch.as_tensor(valid_true[0,0,:], dtype=torch.float64).unsqueeze(0)   # 1X2
x0 = torch.tensor([[0.1,0.1]], dtype=torch.float64).unsqueeze(0)
print(x0.shape)
G = torch.eye(2)
# G = torch.randn(2,2)
var = torch.tensor(var, dtype=torch.float64)

Q = 1e-2 * torch.ones(2,dtype=torch.float64)

model_dict = {
    "name": "Chua_circuit",
    "d": 2,
    "tf": tf,
    "t0": t0,
    "x0": x0,
    "hz": 1/dt,
    "var": var,
    "pnoise_cov": Q,
    "G": G,
    "degree": degree,
}
assert isinstance(model_dict, dict)
save_dir = f"./.data/{fs}ms/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model_dict, save_dir + "model.pt")
# save data
os.makedirs(save_dir + "train/", exist_ok=True)
torch.save(train_t, save_dir + "train/time.pt")
torch.save(train_data, save_dir + "train/obs.pt")
torch.save(train_true, save_dir + "train/true_state.pt")

os.makedirs(save_dir + "valid/", exist_ok=True)
torch.save(valid_t, save_dir + "valid/time.pt")
torch.save(valid_data, save_dir + "valid/obs.pt")
torch.save(valid_true, save_dir + "valid/true_state.pt")

os.makedirs(save_dir + "test/", exist_ok=True)
torch.save(test_t, save_dir + "test/time.pt")
torch.save(test_data, save_dir + "test/obs.pt")
torch.save(test_true, save_dir + "test/true_state.pt")


