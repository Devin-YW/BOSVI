import torch

mse_loss_fn = torch.nn.MSELoss()
def rmse(m_pred, y_true):
    #
    # loss = mse_loss_fn(m_pred, y_true).sqrt()
    # loss_dB = 10*torch.log10(loss)
    return (m_pred - y_true).pow(2).mean().sqrt()



def rmse2(m_pred, y_true):
    # size: N X dim
    return torch.sqrt(torch.mean((m_pred - y_true) ** 2, axis=1))