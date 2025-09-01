import os
import torch
from SviKalmanNet.filter import SVI_KalmanNet_Filter
from SviKalmanNet.cost_Func import my_lossFunc
from torch.optim.lr_scheduler import ExponentialLR

save_num = 25
# lr_kalman = 1e-2    # experiment3 : 2e-2
wd_kalman = 0
torch.set_default_dtype(torch.float64)

class Trainer:

    def __init__(self, data_path, save_path, filter: SVI_KalmanNet_Filter, print_num, lr_kalman=None):
        # Example:
        #   data_path = './.data/syntheticNL/train/(true)
        #   data_path
        #   save_path = '(syntheticNL) Split_KalmanNet.pt'
        self.filter = filter
        self.var = filter.var
        self.save_path = save_path
        self.model_name = filter.model_name
        self.print_num = print_num
        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim

        self.loss_best = 1e4
        if lr_kalman is None:
            lr_kalman = 1e-2
        else:
            lr_kalman = lr_kalman

        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()

        self.loss_fn = my_lossFunc()
        self.optimizer = torch.optim.Adam(self.filter.kf_net.parameters(), lr=lr_kalman)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        # cal_num_param = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(cal_num_param(self.dnn.kf_net))

        self.batch_size = 1
        self.train_count = 0
        self.data_idx = 0

        self.data_path = data_path

        self.data_t = torch.load(data_path + 'time.pt')
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_x = torch.load(data_path + 'prestate.pt')
        self.data_cov = torch.load(data_path + 'precov.pt')

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

    def train_batch_joint(self, model_save_path=None):

        self.optimizer.zero_grad()
        # scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0])
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]
            self.data_cov = self.data_cov[shuffle_idx]
        batch_x = self.data_x[self.data_idx: self.data_idx + self.batch_size]
        batch_y = self.data_y[self.data_idx: self.data_idx + self.batch_size]
        batch_cov = self.data_cov[self.data_idx: self.data_idx + self.batch_size]
        x_hat = torch.zeros_like(self.data_x)
        P_hat = torch.zeros_like(self.data_x)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200// 6, gamma=0.9)
        for i in range(self.batch_size):
            self.filter.state_post = batch_x[i, :, 0].reshape((-1, 1))
            self.filter.P_post = batch_cov[i, :, 0].reshape((-1, 1))

            for ii in range(1, self.seq_len):
                self.filter.filtering(batch_x[i, :, ii].reshape((-1, 1)),
                                      batch_cov[i, :, ii].reshape(-1, 1),
                                      batch_y[i, :, ii].reshape((-1, 1)),
                                      )
            x_hat[i] = self.filter.state_history[:, -self.seq_len:]
            P_hat[i] = self.filter.state_history2[:, -self.seq_len:]
            self.filter.reset(clean_history=False)
        H = self.filter.GSSModel.Jacobian_h(x_hat[:, :, 1])
        # batch_cov = P_hat.permute(0, 2, 1)     # batch_size X 200 X 2
        # P_hat = torch.diag_embed(batch_cov)     # A X C X B X B
        P_diag = torch.diag_embed(batch_cov.permute(0, 2, 1))

        R_diag = torch.diag(self.var)     ###################   batch_x = x_pre
        loss = self.loss_fn(H, batch_y, x_hat, batch_x, R_diag, P_diag, model_params=self.filter.kf_net.parameters())

        # loss = self.loss_fn(batch_x[:, :, 1:], x_hat[:,:,1:]) + self.loss_fn(batch_y[:, :, 1:], H @ x_hat[:,:,1:])
        loss.backward(retain_graph=True)

        ## gradient clipping with maximum value 10
        torch.nn.utils.clip_grad_norm_(self.filter.kf_net.parameters(), 10)  # 梯度剪裁
        self.optimizer.step()

        self.data_idx += self.batch_size

        if model_save_path == None:
            svi_model_path = './.model_saved/'
        else:
            svi_model_path = model_save_path
        os.makedirs(os.path.dirname(svi_model_path), exist_ok=True)
        if self.train_count % self.print_num == 0:
            try:
                print(loss.item())
                torch.save(self.filter.kf_net, svi_model_path + self.save_path[:-3] + '_' + str(self.train_count) + '.pt')
            except:
                print('here')
                pass
        # if self.train_count % print_num == 1:
        #     print(f'[Model {self.save_path}] [Train {self.train_count}] loss [dB] = {10 * torch.log10(loss):.4f}')
        self.train_count += 1
    def validate(self, tester):
        if tester.loss2.item() < self.loss_best:
            try:
                torch.save(tester.filter.kf_net, './.model_saved/' + self.save_path[:-3] + '_best.pt')
                print(f'Save best model at {self.save_path} & train {self.train_count} & loss [dB] = {tester.loss:.4f}')
                self.loss_best = tester.loss2.item()
            except:
                pass
        self.valid_loss = tester.loss2.item()

