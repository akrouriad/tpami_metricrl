import torch
import numpy as np
from gaussian_proj import mean_diff


def cweight_mean_proj(w, mean_mat, wq, old_means, old_prec, epsilon):
    means = w.mm(mean_mat)
    mw = mean_diff(means, old_means, old_prec)
    eta = torch.tensor(1.)
    if mw > epsilon + 1e-6:
        w2 = torch.sum(w ** 2, dim=1)
        wq2 = torch.sum(wq ** 2, dim=1)
        w_ratio = torch.mean(torch.clamp(w2 / wq2, min=1.))
        eta = torch.sqrt(epsilon / mw / w_ratio)

    return eta
