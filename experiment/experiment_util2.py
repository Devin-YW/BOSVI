from SviKalmanNet.filter import SVI_KalmanNet_Filter, Particle_Filter
from SviKalmanNet.trainer import Trainer
from SviKalmanNet.tester import Tester, Tester_PF, Tester_for_svi
from SviKalmanNet.model import train_model_config
from torch.optim.lr_scheduler import ExponentialLR
import torch
import time
from NRMSE import rmse
from svise import odes
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchsde import sdeint as torchsdeint
from tqdm import tqdm
tqdm.disable = True
from svise.sde_learning import *
import os
import numpy as np
from SviKalmanNet.ConvergenceMonitor import EarlyStopping
from src.danse import DANSE, train_danse, test_danse
from Test_gmukf_chua import Tester_GM_UKF
torch.set_default_dtype(torch.float64)


if not os.path.exists("./.model_saved2/"):
    os.makedirs("./.model_saved2/")
model_path = "./.model_saved2/"

if not os.path.exists("./.results/"):
    os.makedirs("./.results/")
# Assembling DANSE
logfile_path = "./log/"
os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
modelfile_path = model_path
os.makedirs(os.path.dirname(modelfile_path), exist_ok=True)
tr_log_file_name = "training.log"
tr_logfile_name_with_path = os.path.join(logfile_path, tr_log_file_name)
# loading mode
# fs = 10 / 20
num_epochs = 300
for fs in [10]:
    model = torch.load(f"./.model_saved2/model_{fs}_ms.pt")
    model_name = model["name"]
    x_dim = model["d"]
    tmin = model["t0"]
    tmax = model["tf"]
    x0 = model["x0"]
    df = model["hz"]
    var = model["var"]

    # var = torch.diag(var)

    Q_diag = model["pnoise_cov"]
    G = model["G"]
    degree = model["degree"]

    # loading data
    train_t = torch.load(f"./.data/train_{fs}ms/time.pt")
    train_data = torch.load(f"./.data/train_{fs}ms/obs.pt")
    train_true = torch.load(f"./.data/train_{fs}ms/true_state.pt")

    print(train_t.shape)
    print(train_true.shape)
    print(train_data.shape)
    valid_t = torch.load(f"./.data/valid_{fs}ms/time.pt")
    valid_data = torch.load(f"./.data/valid_{fs}ms/obs.pt")
    valid_true = torch.load(f"./.data/valid_{fs}ms/true_state.pt")
    mini_batch_t = valid_t.squeeze(0)
    mini_batch_y = valid_data.squeeze(0)

    # degree = 5
    n_reparam_samples = 32
    batch_t = train_t.squeeze(0)
    batch_y = train_data.squeeze(0)
    batch_true = train_true.squeeze(0)

    number = 2000
    batch_t1 = batch_t[:number]
    batch_y1 = batch_y[:number]
    batch_true1 = batch_true[:number]
    batch_t2 = batch_t[number:]
    batch_y2 = batch_y[number:, :]
    batch_true2 = batch_true[number:, :]

    G = torch.as_tensor(G, dtype=torch.float64)
    num_data = len(batch_t1)

    model_config = train_model_config(model_name, x_dim, G.shape[0], G, x0, Q_diag, var, num_data, 1)

    sde2 = SparsePolynomialSDE(
        x_dim,
        (batch_t.min(), batch_t.max()),
        degree=5,
        n_reparam_samples=n_reparam_samples,
        G=G,
        num_meas=G.shape[0],
        measurement_noise=var,
        train_t=batch_t,
        train_x=batch_y,
    )
    sde2.train()
    num_data1 = len(batch_t)
    num_mc_samples = int(min(128, num_data1))

    lr2 = 1e-2  # 20ms: 1e-2  #10ms :3e-2
    # optimizer2 = torch.optim.AdamW(sde2.parameters(), lr=lr2)
    optimizer2 = torch.optim.Adam(
            [{"params": sde2.state_params()}, {"params": sde2.sde_params(), "lr": 1e-2}], lr=lr2
        )
    scheduler2 = lr_scheduler.ExponentialLR(optimizer2, gamma=0.9)
    svi_rng = torch.Generator()
    svi_rng.manual_seed(0)
    train_dataset = TensorDataset(batch_t, batch_y)
    train_loader2 = DataLoader(train_dataset, batch_size=num_mc_samples, shuffle=True)

    t0 = float(batch_t1.min())
    tf = float(batch_t1.max())

    sde = SparsePolynomialSDE(
        x_dim,
        (t0, tf),
        degree=5,
        n_reparam_samples=n_reparam_samples,
        G=G,
        num_meas=G.shape[0],
        measurement_noise=var,
        train_t=batch_t1,
        train_x=batch_y1,
    )
    sde.train()

    num_mc_samples1 = int(min(128, num_data))

    lr = 1e-2   # 20ms: 1e-2 10ms:3e-2

    optimizer = torch.optim.AdamW(sde.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    start_time = time.time()

    bosvi_rng = torch.Generator()
    bosvi_rng.manual_seed(0)
    train_dataset1 = TensorDataset(batch_t1, batch_y1)
    train_loader1 = DataLoader(train_dataset1, batch_size=num_mc_samples1, shuffle=True)
    val_dataset = TensorDataset(mini_batch_t, mini_batch_y)
    val_loader = DataLoader(val_dataset, batch_size=num_mc_samples, shuffle=True)

    print_num = 10
    transition_iters = num_epochs / 2
    scheduler_freq = transition_iters // 2
    SVI_loss = []
    BOSVI_loss = []
    danse_loss = []
    pf_loss = []

    early_stopping = EarlyStopping(tol=1e-4, patience=10)

    mode = "train"
    if mode == "train":
        j = 0
        val_loss_epoch_sum = 0.0
        for epoch in range(num_epochs+1):
            j += 1
            beta = min(1.0, (1.0 * j) / (transition_iters))
            for t_batch, y_batch in train_loader1:
                optimizer.zero_grad()
                if j % scheduler_freq == 0:
                    scheduler.step()
                loss = -sde.elbo(t_batch, y_batch, beta, num_data)
                loss.backward()
                val_loss_epoch_sum += loss.item()
                optimizer.step()
            if epoch > (num_epochs // 1.55):
                val_loss = val_loss_epoch_sum / len(val_loader)
                torch.save(sde, model_path + "pretrain_SVI_deg3.pt")
                early_stopping(val_loss)
                print(f"Epoch {epoch}: Validation Loss = {val_loss}")
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
        #
        # # svi_path = "./.model_saved/SVI_" + str(num_epochs) + ".pt"
        svi_path2 = "./.model_saved2/pretrain_SVI_deg3.pt"
        model_config.generate_for_state(svi_path2, batch_t2, batch_y2, batch_true2)

        data_path = './.data/Chua_circuit/train_state/'

        save_path = 'Svi_KalmanNet.pt'
        filter = SVI_KalmanNet_Filter(model_config)
        trainer_Skalman = Trainer(data_path, save_path, filter, print_num)
        # start training proposed algorithm
        trainer_Skalman.scheduler = ExponentialLR(trainer_Skalman.optimizer, gamma=0.9)

        for epoch in tqdm(range(num_epochs+1)):
            trainer_Skalman.train_batch_joint(model_path)
            if epoch > 110:
                trainer_Skalman.scheduler.step()
            trainer_Skalman.filter.reset(clean_history=True)
            if epoch % print_num == 0:
                with torch.no_grad():
                    BOSVI_tester = Tester(
                        filter=SVI_KalmanNet_Filter(model_config),
                        data_path=f"./.data/valid_{fs}ms/",
                        svi_path=svi_path2,
                        kalman_model=model_path + 'Svi_KalmanNet_' + str(epoch) + '.pt',
                        is_validation=True
                    )
                    BOSVI_loss += [BOSVI_tester.loss2.item()]

        n = 0
        for epoch in range(num_epochs + 1):
            n += 1
            beta = min(1.0, (1.0 * n) / (transition_iters))
            for t_batch, y_batch in train_loader2:
                optimizer2.zero_grad()
                if n % scheduler_freq == 0:
                    scheduler2.step()
                loss = -sde2.elbo(t_batch, y_batch, beta, num_data1)
                loss.backward()
                optimizer2.step()
            if epoch % print_num == 0:
                sde2.eval()
                mu = sde2.marginal_sde.mean(valid_t)
                diag_var = sde2.marginal_sde.K(valid_t).diagonal(dim1=-2, dim2=-1)
                loss_svi = rmse(mu, valid_true)
                loss_dB = 10*torch.log10(loss_svi)
                print(f"{epoch}_total loss (RMSE){loss_dB.item(): .4f}")
                torch.save(sde2, model_path + "SVI_" + str(epoch) + ".pt")
                SVI_loss += [loss_dB.item()]
                sde2.train()
        svi_path = model_path + "SVI_" + str(num_epochs) + ".pt"

        ################################### DANSE ####################################################
        # load GPU IF
        ngpu = 1  # Comment this out if you want to run on cpu and the next line just set device to "cpu"
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        print("Device Used:{}".format(device))
        lr_danse = 3e-3    # 10ms 3e-3 # 20ms: 5e-3
        n_hidden = 20
        danse_rng = torch.Generator()
        danse_rng.manual_seed(2)
        est_parameters_dict = model_config.get_experiment(num_epochs, lr_danse, n_hidden)
        train_loader = DataLoader(TensorDataset(train_true.clone(), train_data.clone()), batch_size=train_data.shape[0],
                                  shuffle=True, generator=danse_rng)
        valid_loader = DataLoader(TensorDataset(valid_true.clone(), valid_data.clone()), batch_size=valid_data.shape[0],
                                  shuffle=True, generator=danse_rng)
        estimator_options = est_parameters_dict["danse"]

        model_danse = DANSE(**estimator_options)
        tr_verbose = True
        # Starting model training
        tr_losses, val_mse_loss, danse_loss, _, _ = train_danse(
            model=model_danse,
            train_loader=train_loader,
            val_loader=valid_loader,
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
            data_path=f"./.data/valid_{fs}ms/",
        )
        estimate3 = tester_pf.est_mean
        pf_loss = [tester_pf.loss2.item()]
        print(pf_loss)

        test_GM_UKF = Tester_GM_UKF("Chua_circuit", model_path=f"./.model_saved2/model_{fs}_ms.pt",
                                    data_path=f"./.data/valid_{fs}ms/")
        X_est, P_est, _, GM_UKF_loss2 = test_GM_UKF.forward()
        gm_ukf_loss = [GM_UKF_loss2.item()]

    # SVI_model = torch.load(svi_path)
    # BOSVI_model = torch.load(model_path + 'Svi_KalmanNet_' + str(num_epochs) + '.pt')
    # torch.save(BOSVI_model, model_path + f"bosvi_model_{fs}ms.pt")
    # DANSE_model = torch.load(model_path + 'danse_gru_ckpt_epoch_' + str(num_epochs) + ".pt")
    # test_t = torch.load(f"./.data/test_{fs}ms/time.pt")
    # test_data = torch.load(f"./.data/test_{fs}ms/obs.pt")
    # test_true = torch.load(f"./.data/test_{fs}ms/true_state.pt")
    # test_loader = DataLoader(TensorDataset(test_true, test_data), batch_size=test_t.shape[0] + 1, shuffle=True)
    # Bo_SVI = model_path + 'Svi_KalmanNet_' + str(num_epochs) + '.pt'
    # Danse_path = "./.model_saved2/danse_gru_ckpt_epoch_" + str(num_epochs) + ".pt"

    # np.save(f'./.results/{fs}ms/demo2/valid_loss_SVI.npy', np.array(SVI_loss))
    # np.save(f'./.results/{fs}ms/demo2/valid_loss_BOSVI.npy', np.array(BOSVI_loss))
    # np.save(f'./.results/{fs}ms/demo2/valid_loss_DANSE.npy', np.array(danse_loss))
    # np.save(f'./.results/{fs}ms/demo2/valid_loss_PF.npy', np.array(pf_loss))
    # np.save(f'./.results/{fs}ms/demo2/valid_loss_GM_UKF.npy', np.array(gm_ukf_loss))

    # torch.save(SVI_model,   model_path + f"svi_model_{fs}ms.pt")
    # torch.save(BOSVI_model, model_path + f"bosvi_model_{fs}ms.pt")
    # torch.save(DANSE_model, model_path + f"danse_model_{fs}ms.pt")