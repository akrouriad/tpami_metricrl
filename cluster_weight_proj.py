import torch
import numpy as np
from gaussian_proj import mean_diff


def cweight_mean_proj(w, means, wq, old_means, old_prec, epsilon):
    mw = mean_diff(means, old_means, old_prec)
    eta = torch.tensor(1.)
    if mw > epsilon + 1e-6:
        nw = torch.sum(w, dim=1)
        nwq = torch.sum(wq, dim=1)
        nw2 = nw ** 2
        nwq2 = nwq ** 2
        w_ratio = torch.clamp(nw2 / nwq2, min=1.)
        eta1 = torch.sqrt(epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio))
        w_ratio = nw2 / (nwq2 + 2 * nwq * (nw - nwq))
        eta2 = epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio)
        eta = torch.max(eta1, eta2)
    return eta


def ls_cweight_mean_proj(w, means, wq, old_means, old_prec, epsilon, cmeans, lstimes=10):
    mw = mean_diff(means, old_means, old_prec)
    eta = torch.tensor(1.)
    if mw > epsilon + 1e-6:
        nw = torch.sum(w, dim=1)
        nwq = torch.sum(wq, dim=1)
        nw2 = nw ** 2
        nwq2 = nwq ** 2
        w_ratio = torch.clamp(nw2 / nwq2, min=1.)
        eta1 = torch.sqrt(epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio))
        w_ratio = nw2 / (nwq2 + 2 * nwq * (nw - nwq))
        eta2 = epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio)
        eta = torch.max(eta1, eta2)

        lb = eta
        ub = 1.
        for keta in range(lstimes):
            ceta = (lb + ub) / 2
            etaw = (1 - ceta) * wq + ceta * w
            etaw /= torch.sum(etaw, dim=1, keepdim=True)
            if mean_diff(etaw.mm(cmeans), old_means, old_prec) < epsilon:
                lb = ceta
            else:
                ub = ceta
        eta = lb
    return eta


def ls_hardcweight_mean_proj(hardning_fc, c, exp_dist, cq, old_means, old_prec, epsilon, cmeans, lstimes=10):
    means = hardning_fc(exp_dist, c).mm(cmeans)
    mw = mean_diff(means, old_means, old_prec)
    eta = torch.tensor(1.)
    if mw > epsilon + 1e-6:
        lb = 0.
        ub = 1.
        for keta in range(lstimes):
            deta = (lb + ub) / 2
            ceta = (1 - deta) * cq + deta * c
            means = hardning_fc(exp_dist, ceta).mm(cmeans)
            if mean_diff(means, old_means, old_prec) > epsilon + 1e-6:
                ub = deta
            else:
                lb = deta
        eta = lb
    return eta

