import torch
import time
import os
from svise import odes
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from SviKalmanNet.model import train_model_config
from SviKalmanNet.filter import Particle_Filter, SVI_KalmanNet_Filter
from torch.utils.data import DataLoader, TensorDataset
from SviKalmanNet.tester import Tester, Tester_PF, Tester_for_svi
from src.danse import test_danse
from Test_gmukf_chua import Tester_GM_UKF
epoch = 300
fs = 10

def db_to_linear(dB_values):
    return 10 ** (np.array(dB_values) / 10)
# loading mode
model = torch.load(f"./.model_saved2/model_{fs}_ms.pt")
model_name = model["name"]
x_dim = model["d"]
tmin = model["t0"]
tmax = model["tf"]
x0 = model["x0"]
df = model["hz"]
var = model["var"]
var = var.unsqueeze(0)
Q_diag = model["pnoise_cov"]
G = model["G"]
degree = model["degree"]
degree = 5
n_reparam_samples = 32
# loading data
train_t = torch.load("./.data/train/time.pt")
batch_t = train_t.squeeze(0)
G = torch.as_tensor(G, dtype=torch.float64)
t0 = float(batch_t.min())
tf = float(batch_t.max())
num_data = len(batch_t)
model_config = train_model_config(model_name, x_dim, G.shape[0], G, x0, Q_diag, var, num_data, 1)

test_t = torch.load(f"./.data/test_{fs}ms/time.pt")
test_data = torch.load(f"./.data/test_{fs}ms/obs.pt")
test_true = torch.load(f"./.data/test_{fs}ms/true_state.pt")
test_loader = DataLoader(TensorDataset(test_true, test_data), batch_size=test_t.shape[0] + 1, shuffle=True)
est_parameters_dict = model_config.get_experiment(300, 1e-3, 20)
estimator_options = est_parameters_dict["danse"]
num_mc_samples = int(min(128, num_data))

# svi_path = f"./.model_saved2/svi_model_{fs}ms.pt"
# BO_svi_path = f"./.model_saved2/bosvi_model_{fs}ms.pt"
# Danse_path = f"./.model_saved2/danse_model_{fs}ms.pt"

#
# svi = torch.load(svi_path)
# bosvi = torch.load(BO_svi_path)
# presvi = torch.load(svi_path2)
# danse = torch.load(Danse_path)
# # #
# torch.save(svi, f"./.results/{fs}ms/demo2/SVI_model.pt")
# torch.save(bosvi, f"./.results/{fs}ms/demo2/BOSVI_model.pt")
# torch.save(presvi, f"./.results/{fs}ms/demo2/PreSVI_model.pt")
# torch.save(danse, f"./.results/{fs}ms/demo2/DANSE_model.pt")

svi_path = f"./.results/{fs}ms/demo2/SVI_model.pt"
svi_path2 = f"./.results/{fs}ms/demo2/PreSVI_model.pt"
BO_svi_path = f"./.results/{fs}ms/demo2/BOSVI_model.pt"
Danse_path = f"./.results/{fs}ms/demo2/DANSE_model.pt"
#
mode = "test"
if mode == "test":
    start = time.perf_counter()
    tester_svi = Tester_for_svi(
        data_path=f"./.data/test_{fs}ms/",
        svi_path=svi_path,
        is_validation=False
    )
    end = time.perf_counter()
    runtime_svi = end - start
    print(f"runtime_svi: {runtime_svi:.6f} s")
    svi_mean = tester_svi.est_mean
    svi_var = tester_svi.est_var
    test_loss_svi = tester_svi.loss
    test_loss_svi2 = tester_svi.loss2

    start = time.perf_counter()
    tester_svi_kf = Tester(
        filter=SVI_KalmanNet_Filter(model_config),
        data_path=f"./.data/test_{fs}ms/",
        svi_path=svi_path2,
        kalman_model=BO_svi_path,
    )
    end = time.perf_counter()
    runtime_bosvi = end - start
    print(f"runtime_bosvi: {runtime_bosvi:.6f} s")
    bosvi_mean = tester_svi_kf.est_mean
    bosvi_var = tester_svi_kf.est_var
    test_loss_bo_svi = tester_svi_kf.loss
    test_loss_bo_svi2 = tester_svi_kf.loss2

    start = time.perf_counter()
    test_loss, test_loss2, est_mean, est_var = test_danse(
        test_loader=test_loader,
        options=estimator_options,
        device="cpu",
        model_file=Danse_path,
        test_logfile_path="./log/testing.log"
    )
    end = time.perf_counter()
    runtime_danse = end - start
    print(f"runtime_danse: {runtime_danse:.6f} s")
    # test_loss_danse = [te_loss.item()]
    test_loss_danse = test_loss
    test_loss_danse2 = test_loss2
    danse_mean = est_mean
    danse_var = est_var

    start = time.perf_counter()
    tester_pf = Tester_PF(
        filter=Particle_Filter(model_config),
        data_path=f"./.data/test_{fs}ms/",
    )
    end = time.perf_counter()
    runtime_pf = end - start
    print(f"runtime_pf: {runtime_pf:.6f} s")
    pf_mean = tester_pf.est_mean
    pf_var = tester_pf.est_var
    test_loss_pf = tester_pf.loss
    test_loss_pf2 = tester_pf.loss2


    test_GM_UKF = Tester_GM_UKF("Chua_circuit", model_path=f"./.model_saved2/model_{fs}_ms.pt",
                             data_path=f"./.data/test_{fs}ms/")
    start = time.perf_counter()
    X_est, P_est, test_loss_gmukf, test_loss_gmukf2 = test_GM_UKF.forward()
    end = time.perf_counter()
    runtime_gmukf = end - start
    print(f"runtime_gmukf: {runtime_gmukf:.6f} s")
    gmukf_mean = X_est
    gmukf_var = P_est

ARMSE = [test_loss_svi2,  test_loss_bo_svi2, test_loss_danse2, test_loss_pf2, test_loss_gmukf2]
loss_values = db_to_linear(ARMSE)
print(loss_values)

labels = ["SVI", "BOSVI", "DANSE", "PF", "GM_UKF"]
errors_db = {
    "SVI": test_loss_svi2.detach().cpu().numpy(),
    "BOSVI": test_loss_bo_svi2.detach().cpu().numpy(),
    "DANSE": test_loss_danse2.detach().cpu().numpy(),
    "PF": test_loss_pf2.detach().cpu().numpy(),
    "GM_UKF": test_loss_gmukf2,
}

pred_vars = {
    "SVI": svi_var.mean().item(),
    "BOSVI": bosvi_var.mean().item(),
    "DANSE": danse_var.mean().item(),
    "PF": pf_var.mean().item(),
    "GM_UKF": gmukf_var.mean().item(),
}

print("==== RMSE 均值 ± 95% CI (from predicted variance) ====")
for label in errors_db.keys():
    mean_val = db_to_linear(errors_db[label])
    ci95 = 1.96 * np.sqrt(pred_vars[label])
    print(f"{label}: {mean_val:.3f} ± {ci95:.3f}")
#10
# SVI: 0.268 ± 0.481
# BOSVI: 0.199 ± 0.002
# DANSE: 0.296 ± 1.012
# PF: 0.891 ± 0.031
# GM_UKF: 0.434 ± 3.795
# 20
# SVI: 0.284 ± 0.524
# BOSVI: 0.215 ± 0.002
# DANSE: 0.305 ± 1.012
# PF: 0.834 ± 0.030
# GM_UKF: 0.415 ± 3.795

import matplotlib.pyplot as plt
import numpy as np

# 数据准备
time_points = ['10ms', '20ms']
categories = ['SVISE', 'BOSVI', 'DANSE', 'PF', 'GM_UKF']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#555555']

data = {
    'SVISE': [0.268, 0.284],
    'BOSVI': [0.199, 0.215],
    'DANSE': [0.296, 0.305],
    'PF': [0.891, 0.834],
    'GM_UKF': [0.434, 0.415]
}

fig, ax = plt.subplots(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


x = np.arange(len(time_points))  # [0,1]
num_categories = len(data)
width = 0.15  # 每个柱子的宽度

# 自动计算 offsets，使柱子均匀分布
offsets = np.linspace(-width*num_categories/2 + width/2, width*num_categories/2 - width/2, num_categories)

# 绘制柱状图
for i, (category, values) in enumerate(data.items()):
    ax.bar(x + offsets[i],
           values,
           width=width,
           label=category,
           color=colors[i % len(colors)],
           edgecolor='black',
           linewidth=0.5)

# 坐标轴装饰
ax.set_ylabel('RMSE', fontsize=15)
ax.set_xlabel('Sampling Rate', fontsize=15)
ax.set_xticks(x)
ax.set_ylim(0, 0.9)
ax.set_xticklabels(time_points)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend(loc='upper center',
         bbox_to_anchor=(0.6, 0.95),
         ncol=2,
         frameon=False,
          fontsize=12)

plt.tight_layout()
# plt.savefig("C:/Users/86135/Desktop/S3_res2.eps", format="eps")
plt.show()
#

data = np.linspace(1, 301, 31, dtype=int)


loss_svi = np.load(f'./.results/{fs}ms/demo2/valid_loss_SVI.npy')

loss_danse = np.load(f'./.results/{fs}ms/demo2/valid_loss_DANSE.npy')

loss_BOSVI = np.load(f'./.results/{fs}ms/demo2/valid_loss_BOSVI.npy')

loss_pf = np.load(f'./.results/{fs}ms/demo2/valid_loss_PF.npy')

loss_gmukf = np.load(f'./.results/{fs}ms/demo2/valid_loss_GM_UKF.npy')

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(data, loss_svi, label='SVISE', linewidth=2.5,
        color='#1f77b4', linestyle='--', marker='*')
# #
ax.plot(data, loss_BOSVI, label='BOSVI', linewidth=2.5,
        color='#d62728', linestyle='--', marker='^')
#
ax.plot(data, loss_danse, label='DANSE', linewidth=2.5,
        color='#2ca02c', linestyle='--', marker='x')

ax.plot(data, loss_pf * np.ones(data.shape), label='PF',
        linewidth=2.5, color='#ff7f0e', linestyle='--', marker='s')

ax.plot(data, loss_gmukf * np.ones(data.shape), label='GM_UKF',
        linewidth=2.5, color='k', linestyle='--', marker='s')

ax.set_xlim(-2, 305)

ax.set_ylim(-7.5, 0)


ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

ax.set_xlabel('Epochs', fontsize=15)
ax.set_ylabel('Validation Loss', fontsize=15)

# ax.set_title('Validation Loss over Epochs', fontsize=16)

ax.tick_params(axis='both', labelsize=13)

ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
# ax.legend(fontsize=12, loc='best', frameon=True)
ax.legend(
    fontsize=13,
    loc='upper center',
    bbox_to_anchor=(0.90, 0.85),
    frameon=True,
    borderaxespad=0.1,
    ncol=1
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
# plt.savefig("C:/Users/86135/Desktop/S3_res2.eps", format="eps", bbox_inches='tight')
plt.show()