import torch


def mean_diff(mu, old_mu, old_prec):
    mud = mu - old_mu
    return cross_mul(mud, old_prec, mud)


def entropy_diff(chol, old_logdetcov):
    return .5 * (old_logdetcov - 2 * torch.sum(torch.log(torch.diag(chol))))


def rot_diff(cov, old_prec):
    dim = cov.size()[0]
    return .5 * (torch.sum(old_prec * cov) - dim)


def cross_mul(l, mat, r):
    return .5 * torch.mean(torch.sum(torch.sum(l[:, :, None] * mat, dim=1) * r, dim=1, keepdim=True))


def utils_from_chol(chol):
    cov = chol.mm(chol.t())
    prec = chol.cholesky_inverse()
    logdetcov = 2 * torch.sum(torch.log(torch.diag(chol)))
    return {'cov': cov, 'prec': prec, 'logdetcov': logdetcov}


def gauss_kl_proj(means, chol, old_means, old_cov, old_prec, old_logdetcov, epsilon):
    cov = chol.mm(chol.t())
    m = mean_diff(means, old_means, old_prec)
    r = rot_diff(cov, old_prec)
    e = entropy_diff(chol, old_logdetcov)
    init_kl = r + m + e
    eta_cov = 1.
    eta_mean = 1.

    if init_kl > epsilon + 1e-6:
        eta_cov = epsilon / torch.clamp(r + m + e, 1e-16)
        ncov = (1 - eta_cov) * old_cov + eta_cov * cov
        chol, cov = ncov.cholesky(), ncov
        r, e = rot_diff(cov, old_prec), entropy_diff(chol, old_logdetcov)

    if r + m + e > epsilon + 1e-6:
        eta_mean = torch.sqrt(torch.clamp(epsilon - e - r, 0) / torch.clamp(m, 1e-6))
        means = (1 - eta_mean) * old_means + eta_mean * means
        m = mean_diff(means, old_means, old_prec)

    final_kl = r + m + e

    return {'eta_mean': eta_mean, 'eta_cov': eta_cov, 'means': means, 'chol': chol, 'init_kl': init_kl, 'final_kl': final_kl}
