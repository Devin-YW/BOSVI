import torch
import numpy as np
import torch.nn as nn
from functools import partial

torch.manual_seed(0)
def series_RLC(R, L, C, t, x):
    v_C = x[..., 0]  # 电容电压
    i_L = x[..., 1]  # 电感电流
    i_source = 2 * torch.sin((2 * torch.pi / 3) * t)  # 电流源
    dx = (1 / C) * i_L
    dy = (-1 / (L * C)) * v_C - (R / L) * i_L + (1 / L) * i_source
    return torch.stack([dx, dy], dim=-1)

class Series_RLC_Circuit:
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self) -> None:
        self.ode = partial(series_RLC, 100, 5, 0.1)
        self.diff = torch.ones(1, 2) * 1e-2

    def f(self, t, y):
        return self.ode(t, y)

    def g(self, t, y):
        return self.diff


