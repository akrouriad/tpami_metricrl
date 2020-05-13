import torch
from .gaussian import mean_diff


def cweight_mean_proj(w_default, w, means, wq, old_means, old_prec, epsilon):
    mw = mean_diff(means, old_means, old_prec)
    eta = torch.tensor(1.)
    if mw > epsilon - 1e-6:
        nw = torch.sum(w, dim=1) + w_default
        nwq = torch.sum(wq, dim=1) + w_default
        nw2 = nw ** 2
        nwq2 = nwq ** 2
        w_ratio = torch.clamp(nw2 / nwq2, min=1.)
        eta1 = torch.sqrt(epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio))
        w_ratio = nw2 / (2 * nwq * nw - nwq2)
        if torch.all(w_ratio >= 0):
            eta2 = epsilon / mean_diff(means, old_means, old_prec, s_weights=w_ratio)
        else:
            eta2 = torch.tensor(0.)
        eta = torch.max(eta1, eta2)

    return eta
