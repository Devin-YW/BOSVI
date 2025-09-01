from SviKalmanNet.filter import SVI_KalmanNet_Filter, Particle_Filter
from SviKalmanNet.trainer import Trainer
from SviKalmanNet.tester import Tester, Tester_PF, Tester_for_svi
from SviKalmanNet.model import train_model_config
from torch.optim.lr_scheduler import ExponentialLR
from Test_gmukf_chua import Tester_GM_UKF
import torch
import time
from NRMSE import rmse
from SviKalmanNet.ConvergenceMonitor import EarlyStopping
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchsde import sdeint as torchsdeint
from tqdm import tqdm
tqdm.disable = True
from svise.sde_learning import *
import os
import numpy as np
from src.danse import DANSE, train_danse, test_danse
torch.set_default_dtype(torch.float64)

fs = 10   # 10/ 20
model_save_path = f"./.model_saved/{fs}ms/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists("./.results/"):
    os.makedirs("./.results/")
# loading mode
num_epochs = 300
model = torch.load(f"./.data/{fs}ms/model.pt")
model_name = model["name"]
x_dim = model["d"]
tmin = model["t0"]
tmax = model["tf"]
x0 = model["x0"]
df = model["hz"]
var = model["var"]
var = torch.diag(var)
print(var.shape)
Q_diag = model["pnoise_cov"]
G = model["G"]
degree = model["degree"]
# loading data
train_t = torch.load(f"./.data/{fs}ms/train/time.pt")
train_data = torch.load(f"./.data/{fs}ms/train/obs.pt")
train_true = torch.load(f"./.data/{fs}ms/train/true_state.pt")

print(train_t.shape)
print(train_true.shape)
print(train_data.shape)
valid_t = torch.load(f"./.data/{fs}ms/valid/time.pt")
valid_data = torch.load(f"./.data/{fs}ms/valid/obs.pt")
valid_true = torch.load(f"./.data/{fs}ms/valid/true_state.pt")
valid_dir = f"./.data/{fs}ms/valid/"
print(valid_t.shape)
print(valid_data.shape)
print(valid_true.shape)

n_reparam_samples = 32
batch_t = train_t.squeeze(0)
batch_y = train_data.squeeze(0)

batch_true = train_true.squeeze(0)

mini_batch_t = valid_t.squeeze(0)
mini_batch_y = valid_data.squeeze(0)

number = 2000
batch_t1 = batch_t[:number]
batch_y1 = batch_y[:number]
batch_true1 = batch_true[:number]
batch_t2 = batch_t[number:]
batch_y2 = batch_y[number:, :]
batch_true2 = batch_true[number:, :]

G = torch.as_tensor(G, dtype=torch.float64)
t0 = float(batch_t.min())
tf = float(batch_t.max())
sde = SparsePolynomialSDE(
    x_dim,
    (t0, tf),
    degree=5,
    n_reparam_samples=n_reparam_samples,
    G=G,
    num_meas=G.shape[0],
    measurement_noise=var,
    train_t=batch_t,
    train_x=batch_y,
)
sde.train()
num_data = len(batch_t)
num_mc_samples = int(min(128, num_data))
lr = 6e-2
optimizer = torch.optim.Adam(
    [{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": 1e-1},], lr=lr
)
# optimizer = torch.optim.AdamW(sde.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
start_time = time.time()
train_dataset = TensorDataset(batch_t, batch_y)
train_loader = DataLoader(train_dataset, batch_size=num_mc_samples, shuffle=True)

# Pretrain for BOSVI
lr2 = 6e-2  # 20ms、(1e-2)  10ms: 6e-2
sde2 = SparsePolynomialSDE(
    x_dim,
    (batch_t1.min(), batch_t1.max()),
    degree=5,
    n_reparam_samples=n_reparam_samples,
    G=G,
    num_meas=G.shape[0],
    measurement_noise=var,
    train_t=batch_t1,
    train_x=batch_y1,
)
sde2.train()
num_data2 = len(batch_t1)
model_config = train_model_config(model_name, x_dim, G.shape[0], G, x0, Q_diag, var, num_data2, 1)
num_mc_samples2 = int(min(128, num_data2))
optimizer2 = torch.optim.AdamW(sde2.parameters(), lr=lr2)
scheduler2 = lr_scheduler.ExponentialLR(optimizer2, gamma=0.9)

train_dataset1 = TensorDataset(batch_t1, batch_y1)
train_loader1 = DataLoader(train_dataset1, batch_size=num_mc_samples2, shuffle=True)

val_dataset = TensorDataset(mini_batch_t, mini_batch_y)
val_loader = DataLoader(val_dataset, batch_size=num_mc_samples2, shuffle=True)

logfile_path = "./log/"
os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
modelfile_path = model_save_path
os.makedirs(os.path.dirname(modelfile_path), exist_ok=True)
tr_log_file_name = "training.log"
tr_logfile_name_with_path = os.path.join(logfile_path, tr_log_file_name)
# load GPU IF
ngpu = 0  # Comment this out if you want to run on cpu and the next line just set device to "cpu"
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device Used:{}".format(device))
danse_rng = torch.Generator()
danse_rng.manual_seed(2)
est_parameters_dict = model_config.get_experiment(num_epochs, lr_danse=1e-2)  # 20ms: 1e-3 10ms:1e-2
train_loader_danse = DataLoader(TensorDataset(train_true, train_data),
                                batch_size=train_data.shape[0] + 1, shuffle=True, generator=danse_rng)
valid_loader_danse = DataLoader(TensorDataset(valid_true, valid_data),
                                batch_size=valid_data.shape[0] + 1, shuffle=True,generator=danse_rng)
estimator_options = est_parameters_dict["danse"]

model_danse = DANSE(**estimator_options)
tr_verbose = True

print_num = 10
transition_iters = num_epochs / 2
scheduler_freq1 = 60
scheduler_freq2 = 70
SVI_loss = []
BOSVI_loss = []
danse_loss = []
pf_loss = []
early_stopping = EarlyStopping(tol=1e-4, patience=10)
mode = "train"
if mode == "train":
    # # 1、BOSVI
    i = 0
    for epoch in range(num_epochs + 1):
        i += 1
        val_loss_epoch_sum = 0.0
        for t_batch, y_batch in train_loader1:
            optimizer2.zero_grad()
            beta = min(1.0, (1.0 * i) / (transition_iters))
            if i % scheduler_freq2 == 0:
                scheduler2.step()
            loss = -sde2.elbo(t_batch, y_batch, beta, num_data2)
            val_loss_epoch_sum += loss.item()
            loss.backward()
            optimizer2.step()
        if epoch > (num_epochs / 1.5):
            val_loss = val_loss_epoch_sum / len(val_loader)
            torch.save(sde2, model_save_path + "pretrain_SVI_deg3.pt")
            early_stopping(val_loss)
            print(f"Epoch {epoch}: Validation Loss = {val_loss}")
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    svi_path = model_save_path + "/pretrain_SVI_deg3.pt"
    model_config.generate_for_state(svi_path, batch_t2, batch_y2, batch_true2)

    data_path = './.data/Chua_circuit/train_state/'

    save_path = 'Svi_KalmanNet.pt'
    filter = SVI_KalmanNet_Filter(model_config)
    trainer_Skalman = Trainer(data_path, save_path, filter, print_num)
    # start training proposed algorithm
    trainer_Skalman.scheduler = ExponentialLR(trainer_Skalman.optimizer, gamma=0.9)

    for epoch in tqdm(range(num_epochs + 1)):
        trainer_Skalman.train_batch_joint(model_save_path)
        if epoch > 110:
            trainer_Skalman.scheduler.step()
        trainer_Skalman.filter.reset(clean_history=True)
        if epoch % print_num == 0:
            BOSVI_tester = Tester(
                filter=SVI_KalmanNet_Filter(model_config),
                data_path=valid_dir,
                svi_path=svi_path,
                kalman_model=model_save_path + 'Svi_KalmanNet_' + str(epoch) + '.pt',
                is_validation=True
            )
            BOSVI_loss += [BOSVI_tester.loss2.item()]
    # #
    j = 0
    # 2、SVI
    for epoch in range(num_epochs+1):
        j += 1
        for t_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if j < transition_iters:
                # beta warmup iters
                beta = min(1.0, (1.0 * j) / (transition_iters))
            else:
                beta = 1.0
            if j % scheduler_freq1 == 0:
                scheduler.step()
            loss = -sde.elbo(t_batch, y_batch, beta, num_data)
            loss.backward()
            optimizer.step()
        if epoch % print_num == 0:
            sde.eval()
            mu = sde.marginal_sde.mean(valid_t)
            diag_var = sde.marginal_sde.K(valid_t).diagonal(dim1=-2, dim2=-1)
            loss_svi = rmse(mu, valid_true)
            loss_dB = 10*torch.log10(loss_svi)
            print(f"{epoch}_total loss (NRMSE){ loss_dB.item(): .4f}")
            torch.save(sde, model_save_path + "SVI_" + str(epoch) + ".pt")
            SVI_loss += [loss_dB.item()]
            sde.train()

    # 3、DANSE

    tr_losses, val_mse_loss, danse_loss, _, _ = train_danse(
        model=model_danse,
        train_loader=train_loader_danse,
        val_loader=valid_loader_danse,
        options=estimator_options,
        nepochs=model_danse.rnn.num_epochs,
        logfile_path=tr_logfile_name_with_path,
        modelfile_path=modelfile_path,
        save_chkpoints="some",
        device=device,
        tr_verbose=tr_verbose,
        pri_num=print_num
    )

    tester_pf = Tester_PF(
        filter=Particle_Filter(model_config),
        data_path=valid_dir,
    )
    estimate3 = tester_pf.est_mean
    pf_loss = [tester_pf.loss2.item()]

    test_GM_UKF = Tester_GM_UKF("Chua_circuit", model_path = f"./.data/{fs}ms/model.pt", data_path = valid_dir)
    X_est, P_est,_, GM_UKF_loss2 = test_GM_UKF.forward()
    gm_ukf_loss = [GM_UKF_loss2.item()]

    # print(len(SVI_loss))
    # print(len(BOSVI_loss))
    # print(len(danse_loss))

    # np.save(f'./.results/{fs}ms/demo1/valid_loss_SVI.npy', np.array(SVI_loss))
    # np.save(f'./.results/{fs}ms/demo1/valid_loss_BOSVI.npy', np.array(BOSVI_loss))
    # np.save(f'./.results/{fs}ms/demo1/valid_loss_DANSE.npy', np.array(danse_loss))
    # np.save(f'./.results/{fs}ms/demo1/valid_loss_PF.npy', np.array(pf_loss))
    # np.save(f'./.results/{fs}ms/demo1/valid_loss_GMUKF.npy', np.array(gm_ukf_loss))
