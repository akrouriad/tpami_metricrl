import torch
import numpy as np


def mean_diff(mu, old_mu, old_prec, s_weights=None):
    mud = mu - old_mu
    return cross_mul(mud, old_prec, mud, s_weights)


def entropy_diff(chol, old_logdetcov):
    return .5 * (old_logdetcov - 2 * torch.sum(torch.log(torch.diag(chol))))


def rot_diff(cov, old_prec):
    dim = cov.size()[0]
    return .5 * (torch.sum(old_prec * cov) - dim)


def cross_mul(l, mat, r, s_weights=None):
    if s_weights is None:
        return .5 * torch.mean(torch.sum(torch.sum(l[:, :, None] * mat, dim=1) * r, dim=1, keepdim=True))
    else:
        return .5 * torch.mean(s_weights * torch.sum(torch.sum(l[:, :, None] * mat, dim=1) * r, dim=1))


def utils_from_chol(chol):
    cov = chol.mm(chol.t())
    prec = chol.cholesky_inverse()
    logdetcov = 2 * torch.sum(torch.log(torch.diag(chol)))
    return {'cov': cov, 'prec': prec, 'logdetcov': logdetcov}


def gauss_kl_proj(means, chol, old_means, old_cov, old_prec, old_logdetcov, epsilon):
    # KL violation
    cov = chol.mm(chol.t())
    m = mean_diff(means, old_means, old_prec)
    r = rot_diff(cov, old_prec)
    e = entropy_diff(chol, old_logdetcov)
    init_kl = m + r + e
    eta_cov = torch.tensor(1.)
    eta_mean = torch.tensor(1.)

    if init_kl > epsilon + 1e-6:
        eta_cov = epsilon / init_kl
        ncov = (1 - eta_cov) * old_cov + eta_cov * cov
        chol, cov = ncov.cholesky(), ncov
        r, e = rot_diff(cov, old_prec), entropy_diff(chol, old_logdetcov)

    if m + r + e > epsilon + 1e-6:
        eta_mean = torch.sqrt(torch.clamp(epsilon - e - r, min=0) / torch.clamp(m, min=1e-6))
        means = (1 - eta_mean) * old_means + eta_mean * means
        m = mean_diff(means, old_means, old_prec)

    final_kl = m + r + e

    return {'eta_mean': eta_mean, 'eta_cov': eta_cov, 'means': means, 'chol': chol, 'init_kl': init_kl, 'final_kl': final_kl}


def entropy(chol):
    a_dim = chol.size()[0]
    return a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(torch.log(torch.diag(chol)))


def project_entrop(chol, e_lb):
    ent = entropy(chol)
    if ent < e_lb:
        a_dim = chol.size()[0]
        chol *= torch.exp((e_lb - ent) / a_dim)
    return chol


def lin_gauss_kl_proj(means, chol, intermediate_means, old_means, old_cov, old_prec, old_logdetcov, epsilon, entrop_lb):
    # KL violation
    chol = project_entrop(chol, entrop_lb)
    cov = chol.mm(chol.t())
    m = mean_diff(means, old_means, old_prec)
    r = rot_diff(cov, old_prec)
    e = entropy_diff(chol, old_logdetcov)
    init_kl = m + r + e
    inter_mean_diff = mean_diff(intermediate_means, old_means, old_prec)
    mm = torch.min(m, inter_mean_diff)
    eta_cov = torch.tensor(1.)
    eta_mean = torch.tensor(1.)

    if init_kl > epsilon + 1e-6:
        eta_cov = (epsilon - mm) / (m + r + e - mm)
        cov = (1. - eta_cov) * old_cov + eta_cov * cov
        chol = cov.cholesky()
        r, e = rot_diff(cov, old_prec), entropy_diff(chol, old_logdetcov)

    if m + r + e > epsilon + 1e-6:
        a = mean_diff(means, intermediate_means, old_prec)
        b = cross_mul(means - intermediate_means, old_prec, intermediate_means - old_means)
        c = epsilon - inter_mean_diff - e - r
        eta_mean = (-b + torch.sqrt(torch.clamp(b * b + a * c, min=1e-16))) / torch.clamp(a, min=1e-6)
        means = (1. - eta_mean) * intermediate_means + eta_mean * means
        m = mean_diff(means, old_means, old_prec)

    final_kl = m + r + e

    return {'eta_mean': eta_mean, 'eta_cov': eta_cov, 'means': means, 'chol': chol, 'init_kl': init_kl, 'final_kl': final_kl}
