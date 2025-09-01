import torch
import math
import torch.nn as nn
import torch.nn.functional as F

nGRU = 1
gru_scale_s = 2

class ConvGate(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,  # 关键修改处
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim  # 使用深度可分离卷积
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x shape: (seq_len=1, batch=1, hidden_dim)
        x = x.permute(1, 2, 0)      # 转换为 (batch, hidden_dim, seq_len)
        gate = self.conv(x)         # 卷积操作
        gate = self.sigmoid(gate)
        gate = gate.permute(2, 0, 1) # 恢复为 (seq_len, batch, hidden_dim)
        return gate * x.permute(2, 0, 1)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        #
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

        self.scale_factor = nn.Parameter(torch.ones(1))

        #
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [seq_len, batch_size, hidden_dim]
        """
        # 计算 Query, Key, Value
        Q = self.query_layer(x)  # [seq_len, batch_size, hidden_dim]
        K = self.key_layer(x)    # [seq_len, batch_size, hidden_dim]
        V = self.value_layer(x)  # [seq_len, batch_size, hidden_dim]

        # # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores * (self.scale_factor / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32)))
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        attention_weights = self.softmax(scores)  # [seq_len, seq_len]

        context = torch.matmul(attention_weights, V)  # [seq_len, batch_size, hidden_dim]

        return context.squeeze(0)

class DNN_SKalmanNet(torch.nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # # # For NCLT, SyntheticNL (general)
        # H1 = (x_dim + y_dim) * 2 * 8
        # H2 = (x_dim + y_dim) * 1 * 4

        # # For 2, 3
        H1 = (x_dim + y_dim) * 3 * 7    # snr
        H2 = (x_dim * y_dim) * 1 * 4

        self.input_dim_2 = (self.x_dim) * 2 + self.y_dim * 2

        self.output_dim_2 = (self.x_dim)

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = round(gru_scale_s * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1

        # attention
        self.attention = SelfAttention(self.gru_hidden_dim)
        # self.conv_gate = ConvGate(self.gru_hidden_dim, kernel_size=3)

        # input layer {residual}
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_2, self.gru_input_dim),
            nn.ReLU(),
        )
        # GRU
        self.hn2 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn2_init = self.hn2.detach().clone()
        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)
        # self.GRU2 = nn.LSTM(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2)
        )
        self.l3 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2)
        )

    def initialize_hidden(self):
        # self.hn1 = self.hn1_init.detach().clone()
        self.hn2 = self.hn2_init.detach().clone()

    def forward(self, state_inno, precov, residual, meas_cov):
        # input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), axis=0).reshape(-1)
        input2 = torch.cat((state_inno, precov, residual, meas_cov), dim=0).reshape(-1)

        l1_out = self.l1(input2)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0, 0, :] = l1_out
        GRU_out, self.hn2 = self.GRU2(GRU_in, self.hn2)

        # attention_out = self.attention(GRU_out)

        X_hat = self.l2(GRU_out)        #state

        var_out = self.l3(GRU_out)

        # P_hat = torch.nn.functional.softplus(var_out)  # 确保非负

        x_hat = X_hat.reshape(self.x_dim, 1)
        P_hat = var_out.reshape(self.x_dim, 1)

        return x_hat, P_hat


