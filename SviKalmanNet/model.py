import torch
import os
import numpy as np
from scipy.stats import levy_stable
import math, random
import configparser
from scipy.special import comb
from scipy.interpolate import interp1d
import torchsde
from scipy.integrate import solve_ivp
from SviKalmanNet.odes import Series_RLC_Circuit
from svise import utils

if not os.path.exists('./data'):
    os.mkdir('./data')

torch.manual_seed(0)  # CPU
# torch.cuda.manual_seed(0)  # GPU
std = 1e-2
degree = 5

class train_model_config:

    def __init__(self, model_name, x_dim, y_dim, h, x0, Q_diag, var, num_data=None, num_traj=None):
        self.name = model_name
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x0 = x0.reshape(x_dim, 1)
        self.G = h
        self.num_data = num_data
        # self.batch_t = train_t
        # self.batch_true = train_data
        # self.batch_y = train_y
        self.num_traj = num_traj
        self.Q_diag = Q_diag
        self.var = var  # 1X2


    def h(self):

        return self.G

    def generate_for_state(self, svi_path, batch_t, batch_y, batch_true):  # dnn_model 为学习未知系统模型的model

        model = torch.load(svi_path)
        save_path = "./.data/" + self.name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tempt_path = save_path + 'train_state/'
        if not os.path.exists(tempt_path):
            os.mkdir(tempt_path)
        self.num_data = len(batch_t)
        self.num_traj = 3
        # data_t = batch_t
        # data_true = batch_true
        # data_y = batch_y
        # num_data = len(data_t)
        # x_dim = data_true.shape[1]
        with torch.no_grad():
            mu = torch.zeros(self.num_traj, self.num_data, self.x_dim)
            var = torch.zeros(self.num_traj, self.num_data, self.x_dim)
            true_data = torch.zeros(self.num_traj, self.num_data, self.x_dim)
            observations = torch.zeros(self.num_traj, self.num_data, self.y_dim)
            for i in range(self.num_traj):

                mu[i, :, :] = model.marginal_sde.mean(batch_t)
                var[i, :, :] = model.marginal_sde.K(batch_t).diagonal(dim1=-2, dim2=-1)

                observations[i, :, :] = batch_y
                true_data[i, :, :] = batch_true
                lb = mu - 2 * var.sqrt()
                ub = mu + 2 * var.sqrt()
        # 使用 permute 调整维度顺序 以适应 KalmanNet

        mu = mu.permute(0, 2, 1)
        var = var.permute(0, 2, 1)
        true_data = true_data.permute(0, 2, 1)
        observations = observations.permute(0, 2, 1)

        torch.save(batch_t, tempt_path + 'time.pt')
        torch.save(true_data, tempt_path + "true_data.pt")
        torch.save(observations, tempt_path + 'obs.pt')
        torch.save(mu, tempt_path + 'prestate.pt')
        torch.save(var, tempt_path + 'precov.pt')

    def Jacobian_h(self, x):
        return self.G

    def get_experiment(self, nenpochs, lr_danse=1e-3, n_hidden=10, device="cpu"):  # set for danse
        estimators_dict = {
            # Parameters of the DANSE estimator
            "danse": {
                "n_states": self.x_dim,
                "n_obs": self.y_dim,
                "mu_w": np.zeros((self.y_dim,)),
                "C_w": self.var.numpy() + 1e-3 * np.eye(self.y_dim),
                # experiment3
                # "mu_w": np.random.normal(loc=0, scale=np.sqrt(self.var.numpy()), size=(self.y_dim,)),
                # "C_w": self.var.numpy() + 1e-3 * np.eye(self.y_dim),
                "H":  self.G.numpy(),
                "mu_x0": np.zeros((self.x_dim,)),
                "C_x0": np.eye(self.x_dim, self.x_dim),
                "batch_size": 10,
                "rnn_type": "gru",
                "device": device,
                "rnn_params_dict": {
                    "gru": {
                        "model_type": "gru",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 40,   # 40
                        "n_layers": 2,    # 2
                        "lr": lr_danse,       # 1e-3   (1,2,3.1)  3.3 3e-3
                        "num_epochs": nenpochs,
                        "min_delta": 5e-3,
                        "n_hidden_dense": n_hidden,
                        "device": device
                    },
                    "rnn": {
                        "model_type": "gru",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 30,    # 40
                        "n_layers": 1,
                        "lr": 1e-3,
                        "num_epochs": nenpochs,
                        "min_delta": 1e-3,
                        "n_hidden_dense": 10,
                        "device": device
                    },
                    "lstm": {
                        "model_type": "lstm",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 30,   # 40
                        "n_layers": 1,
                        "lr": 1e-3,
                        "num_epochs": nenpochs,
                        "min_delta": 1e-3,
                        "n_hidden_dense": 10,
                        "device": device
                    }
                }
            },
            "danse_supervised": {
                "n_states": self.x_dim,
                "n_obs": self.y_dim,
                "mu_w": np.zeros((self.y_dim,)),
                "C_w": None,
                "H": None,
                "mu_x0": np.zeros((self.x_dim,)),
                "C_x0": np.eye(self.x_dim, self.x_dim),
                "batch_size": 64,
                "rnn_type": "gru",
                "device": device,
                "rnn_params_dict": {
                    "gru": {
                        "model_type": "gru",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 30,
                        "n_layers": 1,
                        "lr": 5e-3,
                        "num_epochs": 2000,
                        "min_delta": 5e-2,
                        "n_hidden_dense": 32,
                        "device": device
                    },
                    "rnn": {
                        "model_type": "gru",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 40,
                        "n_layers": 2,
                        "lr": 1e-3,
                        "num_epochs": 300,
                        "min_delta": 1e-3,
                        "n_hidden_dense": 32,
                        "device": device
                    },
                    "lstm": {
                        "model_type": "lstm",
                        "input_size": self.y_dim,
                        "output_size": self.x_dim,
                        "n_hidden": 40,
                        "n_layers": 2,
                        "lr": 1e-3,
                        "num_epochs": 300,
                        "min_delta": 1e-3,
                        "n_hidden_dense": 32,
                        "device": device
                    }
                }
            },
            # Parameters of the Model-based filters - KF, EKF, UKF
            "KF": {
                "n_states": self.x_dim,
                "n_obs": self.y_dim
            },
            "EKF": {
                "n_states": self.x_dim,
                "n_obs": self.y_dim
            },
            "UKF": {
                "n_states": self.x_dim,
                "n_obs": self.y_dim,
                "n_sigma": self.x_dim * 2,
                "kappa": 0.0,
                "alpha": 1e-3
            },
            "KNetUoffline": {
                "n_states": self.x_dim,
                "n_obs":self.y_dim,
                "n_layers": 1,
                "N_E": 10_0,
                "N_CV": 100,
                "N_T": 10_0,
                "unsupervised": True,
                "data_file_specification": 'Ratio_{}---R_{}---T_{}',
                "model_file_specification": 'Ratio_{}---R_{}---T_{}---unsupervised_{}',
                "nu_dB": 0.0,
                "lr": 1e-3,
                "weight_decay": 1e-6,
                "num_epochs": 100,
                "batch_size": 100,
                "device": device
            },
            "dmm": {
                "obs_dim": self.y_dim,  # Dimension of the observation / input to RNN
                "latent_dim": self.x_dim,  # Dimension of the latent state / output of RNN in case of state estimation
                "use_mean_field_q": False,  # Flag to indicate the use of mean-field q(x_{1:T} \vert y_{1:T})
                "batch_size": 128,  # Batch size for training
                "rnn_model_type": 'gru',  # Sets the type of RNN
                "inference_mode": 'st-l',
                # String to indicate the type of DMM inference mode (typically, we will use ST-L or MF-L)
                "combiner_dim": 40,  # Dimension of hidden layer of combiner network
                "train_emission": False,
                # Flag to indicate if emission network needs to be learned (True) or not (False)
                "H": None,  # Measurement matrix, in case of nontrainable emission network with linear measurement
                "C_w": None,
                # Measurmenet noise cov. matrix, in case of nontrainable emission network with linear measurements
                "emission_dim": 40,  # Dimension of hidden layer for emission network
                "emission_num_layers": 1,  # No. of hidden layers for emission network
                "emission_use_binary_obs": False,  # Flag to indicate the modeling of binary observations or not
                "train_transition": True,
                # Flag to indicate if transition network needs to be learned (True) or not (False)
                "transition_dim": 40,  # Dimension of hidden layer for transition network
                "transition_num_layers": 2,  # No. of hidden layers for transition network
                "train_initials": False,  # Set if the initial states also are learned uring the optimization
                "device": device,
                "rnn_params_dict": {
                    "gru": {
                        "model_type": "gru",  # Type of RNN used (GRU / LSTM / RNN)
                        "input_size": self.y_dim,  # Input size of the RNN
                        "output_size": self.x_dim,  # Output size of the RNN
                        "batch_first": True,
                        # Flag to indicate the input tensor is of the form (batch_size x seq_length x input_dim)
                        "bias": True,  # Use bias in RNNs
                        "n_hidden": 40,  # Dimension of the RNN latent space
                        "n_layers": 1,  # No. of RNN layers
                        "bidirectional": False,  # In case of using DKS say then set this to True
                        "dropout": 0.0,  # In case of using RNN dropout
                        "device": device  # Device to set the RNN
                    },
                },
                "optimizer_params": {
                    "type": "Adam",
                    "args": {
                        "lr": 5e-3,  # Learning rate
                        "weight_decay": 0.0,  # Weight decay
                        "amsgrad": True,  # AMS Grad mode to be used for RNN
                        "betas": [0.9, 0.999]  # Betas for Adam
                    },
                    "num_epochs": 2000,  # Number of epochs
                    "min_delta": 5e-3,  # Sets the delta to control the early stopping
                },
            }
        }
        return estimators_dict


