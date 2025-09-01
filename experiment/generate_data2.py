import torch.multiprocessing.reductions
from scipy.io import loadmat
import os
import numpy as np
from scipy.stats import levy_stable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from SviKalmanNet.low_rank_obs import low_rank_obs_matrix
from sklearn.preprocessing import StandardScaler
torch.set_default_dtype(torch.float64)

# G = low_rank_obs_matrix(x_dim=2, r=1)
# G = torch.eye(2)
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

degree = 5

for fs in [10, 20]:
    data = loadmat(f'Chua_6dB_{fs}ms.mat')
    print(data.keys())
    t = data['t_span'].flatten()
    print(t.shape)
    true_data = data['true_data']
    G = data["G"]
    true_data = true_data.T  # (n, 2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    true_data = scaler.fit_transform(true_data)
    print(true_data.shape)
    # noise_data = data['noise_data']

    y_sparse = true_data @ G.T
    print(y_sparse.shape)
    true_var = np.mean(y_sparse**2)
    noise_var = true_var / (10 ** (6 / 10))
    stable_noise = levy_stable.rvs(1.5, 0, size=y_sparse.shape)
    var = noise_var / np.mean(stable_noise**2)
    noise_data = y_sparse + stable_noise * np.sqrt(var)
    # noise_data = data["noise_data"]
    # noise_data_normalized = scaler.fit_transform(noise_data)

    train_data, valid_data, test_data, _, _, _ = split_dataset(noise_data, t)

    train_true, valid_true, test_true,  train_t, valid_t, test_t = split_dataset(true_data, t)

    print("归一化后的数据形状：", true_data.shape)
    print("归一化后的数据形状：", noise_data.shape)
    print("训练集数据形状：", train_data.shape)
    print("测试集数据形状：", test_data.shape)
    print(valid_data.shape)
    print(len(train_data) == len(train_true))
    print(len(test_data) == len(test_true))
    print(len(train_data) == len(train_t))

    '''# 6. 可视化数据
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

    train_t = torch.tensor(train_t,  dtype=torch.float64).unsqueeze(0)
    train_data = torch.tensor(train_data, dtype=torch.float64).unsqueeze(0)
    train_true = torch.tensor(train_true, dtype=torch.float64).unsqueeze(0)
    length = train_t.shape[1]

    valid_t = torch.tensor(valid_t,  dtype=torch.float64).unsqueeze(0)
    valid_data = torch.tensor(valid_data, dtype=torch.float64).unsqueeze(0)
    valid_true = torch.tensor(valid_true, dtype=torch.float64).unsqueeze(0)

    test_t = torch.tensor(test_t,  dtype=torch.float64).unsqueeze(0)
    test_data = torch.tensor(test_data, dtype=torch.float64).unsqueeze(0)
    test_true = torch.tensor(test_true, dtype=torch.float64).unsqueeze(0)

    dt = 0.01
    t0 = float(train_t.min())
    tf = float(train_t.max())
    # 1X2
    x0 = torch.tensor([[0.1, 0.1]], dtype=torch.float64).unsqueeze(0)
    print(x0.shape)

    var = torch.tensor(var, dtype=torch.float64)
    var = var.unsqueeze(0)
    G = torch.tensor(G, dtype=torch.float64)
    Q = 1e-2 * torch.ones(2, dtype=torch.float64)

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

    if not os.path.exists("./.model_saved2/"):
        os.makedirs("./.model_saved2/")
    torch.save(model_dict, f"./.model_saved2/model_{fs}_ms.pt")
    # save data
    os.makedirs(f"./.data/train_{fs}ms/", exist_ok=True)
    torch.save(train_t, f"./.data/train_{fs}ms/time.pt")
    torch.save(train_data, f"./.data/train_{fs}ms/obs.pt")
    torch.save(train_true, f"./.data/train_{fs}ms/true_state.pt")

    os.makedirs(f"./.data/valid_{fs}ms/", exist_ok=True)
    torch.save(valid_t, f"./.data/valid_{fs}ms/time.pt")
    torch.save(valid_data, f"./.data/valid_{fs}ms/obs.pt")
    torch.save(valid_true, f"./.data/valid_{fs}ms/true_state.pt")

    os.makedirs(f"./.data/test_{fs}ms/", exist_ok=True)
    torch.save(test_t, f"./.data/test_{fs}ms/time.pt")
    torch.save(test_data, f"./.data/test_{fs}ms/obs.pt")
    torch.save(test_true, f"./.data/test_{fs}ms/true_state.pt")
