import math
import torch
from SviKalmanNet.filter import SVI_KalmanNet_Filter, Particle_Filter
import time
print_num = 25
class Tester():
    def __init__(self, filter, data_path, svi_path, kalman_model, is_validation=False, is_mismatch=False):
        # Example:
        #   data_path  = './.data/syntheticNL/test/(true)
        #   sys_model = './.model_saved/(PRLCC) SVI_' + str(epoch) + '.pt'
        #   model_path = ./.model_saved/(PRLCC) Svi_KalmanNet_self.count.pt'
        # self.sys_model = torch.load(svi_path)
        self.sys_model = svi_path
        if isinstance(filter, SVI_KalmanNet_Filter):
            self.result_path = 'SVI_KF '

        self.filter = filter
        if not isinstance(filter, Particle_Filter):
            self.filter.kf_net = torch.load(kalman_model)
            self.filter.kf_net.initialize_hidden()
        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim

        self.is_validation = is_validation
        self.is_mismatch = is_mismatch

        self.loss_fn = torch.nn.MSELoss()

        # self.data_t = torch.load(data_path + 'time.pt').to(dtype=torch.float64)   # traj X num
        # self.data_y = torch.load(data_path + 'obs.pt').to(dtype=torch.float64).permute(0, 2, 1)    # traj X 2 X num
        # self.data_true = torch.load(data_path + 'true_state.pt').to(dtype=torch.float64).permute(0, 2, 1)
        ####################################
        if is_validation:
            self.data_t = data_path["valid_t"]
            self.data_y = data_path["valid_y"].unsqueeze(0).permute(0,2,1)
            self.data_true = data_path["valid_state"].unsqueeze(0).permute(0,2,1)
        else:
            self.data_t = data_path["train_t"]
            self.data_y = data_path["train_y"].unsqueeze(0).permute(0, 2, 1)
            self.data_true = data_path["train_state"].unsqueeze(0).permute(0, 2, 1)
        # self.sys_model.eval()
        # self.sys_model.sde_prior.reset_sparse_index()   # 重置索引
        # self.sys_model.sde_prior.update_sparse_index()  # 更新索引

        num_traj = self.data_true.shape[0]
        num_data = self.data_true.shape[2]
        x_dim = self.data_true.shape[1]
        mu = torch.zeros(num_traj, num_data, x_dim)
        var = torch.zeros(num_traj, num_data, x_dim)
        true_data = torch.zeros(num_traj, num_data, x_dim)
        observations = torch.zeros(num_traj, num_data, x_dim)
        for i in range(num_traj):
            mu[i, :, :] = self.sys_model.marginal_sde.mean(self.data_t[i])
            var[i, :, :] = self.sys_model.marginal_sde.K(self.data_t[i]).diagonal(dim1=-2, dim2=-1)
#     # get the state estimate
        self.data_x = mu.permute(0,2,1)    # traj X 2 X num
        self.data_cov = var.permute(0,2,1)

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        x_hat = torch.zeros_like(self.data_true)

        start_time = time.monotonic()
        with torch.no_grad():
            for i in range(self.data_num):
                if i % print_num == 0:
                    if self.is_validation:
                        print("Validating SVI_KALMAN")
                        # print(f'Validating {i} / {self.data_num} of {self.model_path}')
                    else:
                        print("Testing SVI_KALMAN")
                        # print(f'Testing {i} / {self.data_num} of {self.model_path}')
                self.filter.state_post = self.data_x[i, :, 0].reshape((-1, 1))
                if isinstance(filter, SVI_KalmanNet_Filter):
                    for ii in range(1, self.seq_len):
                        self.filter.filtering(self.data_x[i, :, ii].reshape((-1, 1)),
                                              self.data_cov[i, :, ii].reshape(-1, 1),
                                              self.data_y[i, :, ii].reshape((-1, 1)))
                if isinstance(filter, Particle_Filter):
                    self.filter.filtering(self.data_t[i],
                                          self.data_y[i].permute(1,0),
                                          self.data_true[i].permute(1,0))
                x_hat[i] = self.filter.state_history[:, -self.seq_len:]
                self.filter.reset(clean_history=False)

            end_time = time.monotonic()
            # print(timedelta(seconds=end_time - start_time))
            # torch.save(x_hat, data_path + self.result_path + 'x_hat.pt')
            if not is_validation:
                self.estimate = x_hat.squeeze(0).permute(1,0)
                var = var.squeeze(0)
                self.lb = self.estimate - 2 * var.sqrt()
                self.ub = self.estimate + 2 * var.sqrt()

            loss = self.loss_fn(self.data_true[:, :, 1:], x_hat[:, :, 1:])
            # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
            loss_dB = 10 * torch.log10(loss)
            print(f'loss [dB] = {loss_dB.item():.4f}')
            self.loss = loss_dB

            # Compute loss at instantaneous time
            self.loss_instant = torch.zeros(self.data_true[:, :, 1:].shape[-1])
            for i in range(self.data_true[:, :, 1:].shape[-1]):
                self.loss_instant[i] = self.loss_fn(self.data_true[:, :, i + 1], x_hat[:, :, i + 1])
            self.loss_instant_dB = 10 * torch.log10(self.loss_instant)



class Tester_for_svi:
    def __init__(self, data_path, model_path, is_validation=False):
        # data_path = './data/PRLCC/valid_data/PRLCC_permille_0100data_01.pt'
        self.test_model = torch.load(model_path)
        data = torch.load(data_path)
        self.data_t = data["time"]
        self.data_true = data["state"]
        self.data_y = data["obs"]

        self.is_validation = is_validation

        self.loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            if self.is_validation:
                print(f'Validating SVI…………')
            else:
                print(f'Testing SVI…………')

            mu = self.test_model.marginal_sde.mean(self.data_t)
            var = self.test_model.marginal_sde.K(self.data_t).diagonal(dim1=-2, dim2=-1)
            lb = mu - 2 * var.sqrt()
            ub = mu + 2 * var.sqrt()

            loss = self.loss_fn(self.data_true[1:, :], mu[1:, :])
            # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
            loss_dB = 10 * torch.log10(loss)
            print(f'loss [dB] = {loss_dB:.4f}')

            # Compute loss at instantaneous time
            # self.loss_instant = torch.zeros(self.data_true[:, 1:].shape[-1])
            # for i in range(self.data_true[:, 1:].shape[-1]):
            #     self.loss_instant[i] = self.loss_fn(self.data_true[:, i + 1], mu[:, i + 1])
            # self.loss_instant_dB = 10 * torch.log10(self.loss_instant)

        self.loss = loss_dB
