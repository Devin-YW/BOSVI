import torch
import os
import time
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

def db_to_linear(dB_values):
    return 10 ** (np.array(dB_values) / 10)

fs = 20  # 10/20
epoch = 300
# loading mode
model = torch.load(f"./.data/{fs}ms/model.pt")
model_name = model["name"]
x_dim = model["d"]
tmin = model["t0"]
tmax = model["tf"]
x0 = model["x0"]
df = model["hz"]
var = model["var"]
var = torch.diag(var)
print(var)
Q_diag = model["pnoise_cov"]
G = model["G"]
degree = model["degree"]
n_reparam_samples = 32
# loading data
train_t = torch.load(f"./.data/{fs}ms/train/time.pt")
batch_t = train_t.squeeze(0)
G = torch.as_tensor(G, dtype=torch.float64)
t0 = float(batch_t.min())
tf = float(batch_t.max())
num_data = len(batch_t)
model_config = train_model_config(model_name, x_dim, G.shape[0], G, x0, Q_diag, var, num_data, 1)

test_t = torch.load(f"./.data/{fs}ms/test/time.pt")
test_data = torch.load(f"./.data/{fs}ms/test/obs.pt")
test_true = torch.load(f"./.data/{fs}ms/test/true_state.pt")
test_loader = DataLoader(TensorDataset(test_true, test_data), batch_size=test_t.shape[0] + 1, shuffle=True)
est_parameters_dict = model_config.get_experiment(300)
estimator_options = est_parameters_dict["danse"]
num_mc_samples = int(min(128, num_data))
#
# svi_path = f"./.model_saved/{fs}ms/SVI_" + str(epoch) + ".pt"
# svi_path2 = f"./.model_saved/{fs}ms/pretrain_SVI_deg3.pt"
# BO_svi_path = f"./.model_saved/{fs}ms/Svi_KalmanNet_" + str(epoch) + ".pt"
# Danse_path = f"./.model_saved/{fs}ms/danse_gru_ckpt_epoch_" + str(epoch) + ".pt"
# # #
# svi = torch.load(svi_path)
# bosvi = torch.load(BO_svi_path)
# presvi = torch.load(svi_path2)
# danse = torch.load(Danse_path)
# # # # #
# torch.save(svi, f"./.results/{fs}ms/demo1/SVI_model.pt")
# torch.save(bosvi, f"./.results/{fs}ms/demo1/BOSVI_model.pt")
# torch.save(presvi, f"./.results/{fs}ms/demo1/PreSVI_model.pt")
# torch.save(danse, f"./.results/{fs}ms/demo1/DANSE_model.pt")

svi_path = f"./.results/{fs}ms/demo1/SVI_model.pt"
svi_path2 = f"./.results/{fs}ms/demo1/PreSVI_model.pt"
BO_svi_path = f"./.results/{fs}ms/demo1/BOSVI_model.pt"
Danse_path = f"./.results/{fs}ms/demo1/DANSE_model.pt"

test_dir = f"./.data/{fs}ms/test/"
mode = "test"
if mode == "test":
    start = time.perf_counter()
    tester_svi = Tester_for_svi(
        data_path=test_dir,
        svi_path=svi_path,
    )
    svi_mean = tester_svi.est_mean
    svi_var = tester_svi.est_var
    test_loss_svi = tester_svi.loss
    test_loss_svi2 = tester_svi.loss2
    end = time.perf_counter()
    runtime_svi = end - start
    print(f"runtime_svi: {runtime_svi:.6f} s")

    start = time.perf_counter()
    tester_svi_kf = Tester(
        filter=SVI_KalmanNet_Filter(model_config),
        data_path=test_dir,
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
    tester_pf = Tester_PF(
        filter=Particle_Filter(model_config),
        data_path=test_dir,
    )
    end = time.perf_counter()
    runtime_pf = end - start
    print(f"runtime_pf: {runtime_pf:.6f} s")
    pf_mean = tester_pf.est_mean
    pf_var = tester_pf.est_var
    test_loss_pf = tester_pf.loss
    test_loss_pf2 = tester_pf.loss2

    start = time.perf_counter()
    test_loss, test_loss2, est_mean, est_var = test_danse(
        test_loader=test_loader,
        options=estimator_options,
        device="cpu",
        model_file=Danse_path,
        test_logfile_path="./log/testing.log"
    )
    # test_loss_danse = [te_loss.item()]
    end = time.perf_counter()
    runtime_danse = end - start
    print(f"runtime_danse: {runtime_danse:.6f} s")
    test_loss_danse = test_loss
    test_loss_danse2 = test_loss2
    danse_mean = est_mean
    danse_var = est_var

    start = time.perf_counter()
    test_GM_UKF = Tester_GM_UKF("Chua_circuit", model_path=f"./.data/{fs}ms/model.pt", data_path=test_dir)
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

# 10ms
# SVI: 0.277 ± 0.440
# BOSVI: 0.263 ± 0.002
# DANSE: 0.310 ± 0.313
# PF: 0.268 ± 0.111
# GM_UKF: 0.255 ± 0.351

#20ms
# SVI: 0.272 ± 0.423
# BOSVI: 0.261 ± 0.002
# DANSE: 0.270 ± 0.432
# PF: 0.357 ± 0.111
# GM_UKF: 0.337 ± 0.352

import matplotlib.pyplot as plt
import numpy as np

time_points = ['10ms', '20ms']
categories = ['SVISE', 'BOSVI', 'DANSE', 'PF', 'GM_UKF']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#555555']

data = {
    'SVISE': [0.277, 0.272],
    'BOSVI': [0.263, 0.261],
    'DANSE': [0.310, 0.270],
    'PF': [0.268, 0.357],
    'GM_UKF': [0.255, 0.337]
}
# #  10ms  [0.27712992 0.26229291 0.31691772 0.28355616]
# #  20ms  [0.27180326 0.26285301 0.2704162  0.36414219]

fig, ax = plt.subplots(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


x = np.arange(len(time_points))  # [0,1]
num_categories = len(data)
width = 0.15

offsets = np.linspace(-width*num_categories/2 + width/2, width*num_categories/2 - width/2, num_categories)

for i, (category, values) in enumerate(data.items()):
    ax.bar(x + offsets[i],
           values,
           width=width,
           label=category,
           color=colors[i % len(colors)],
           edgecolor='black',
           linewidth=0.5)

ax.set_ylabel('RMSE', fontsize=15)
ax.set_xlabel('Sampling Rate', fontsize=15)
ax.set_xticks(x)
ax.set_ylim(0,0.4)
ax.set_xticklabels(time_points)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, 0.99),
          ncol=num_categories,
          frameon=False,
          fontsize=12)

plt.tight_layout()

# plt.savefig("C:/Users/86135/Desktop/S3_res1.eps", format="eps")
# plt.show()

# plot the results
import numpy as np
import matplotlib.pyplot as plt

data = np.linspace(0, 300, 31, dtype=int)

loss_svi = np.load(f'./.results/{fs}ms/demo1/valid_loss_SVI.npy')

loss_danse = np.load(f'./.results/{fs}ms/demo1/valid_loss_DANSE.npy')

loss_BOSVI = np.load(f'./.results/{fs}ms/demo1/valid_loss_BOSVI.npy')

loss_pf = np.load(f'./.results/{fs}ms/demo1/valid_loss_PF.npy')

loss_gm_ukf = np.load(f'./.results/{fs}ms/demo1/valid_loss_GMUKF.npy')

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

ax.plot(data, loss_gm_ukf * np.ones(data.shape), label='GM_UKF',
        linewidth=2.5, color='k', linestyle='--', marker='s')

ax.set_xlim(-2, 305)

ax.set_ylim(-6.1, -2.9)


ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

ax.set_xlabel('Epochs', fontsize=14)
ax.set_ylabel('Validation Loss', fontsize=14)

# ax.set_title('Validation Loss over Epochs', fontsize=16)

ax.tick_params(axis='both', labelsize=12)

ax.set_xticks([1, 50, 100, 150, 200, 250, 300])

ax.legend(fontsize=18, loc='upper right', frameon=True)

# plt.subplots_adjust(right=0.98)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# plt.savefig("C:/Users/86135/Desktop/S3_10ms_demo1_fig1.eps", format="eps", bbox_inches='tight')
plt.tight_layout()
plt.show()