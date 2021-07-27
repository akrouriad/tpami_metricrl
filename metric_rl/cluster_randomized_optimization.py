import numpy as np


class Sampler(object):
    def __init__(self, w):
        self._w = w

    def sample(self, n):
        if n == len(self._w):
            return np.arange(len(self._w))

        return np.random.choice(len(self._w), n, replace=False, p=self._w)

    @staticmethod
    def get_rank(h):
        tmp = np.argsort(h)
        rank = np.empty_like(tmp)
        rank[tmp] = np.arange(len(tmp))[::-1]

        return rank


class PolynomialSampling(Sampler):
    def __init__(self, h, exponent):
        rank = Sampler.get_rank(h)
        w = np.array([1.0/(1.0+i)**exponent for i in rank])

        super().__init__(w/np.sum(w))


class LogarithmicSampling(Sampler):
    def __init__(self, h):
        rank = Sampler.get_rank(h)
        w = np.array([1.0 / (1.0 + np.log(i + 1.0)) for i in rank])

        super().__init__(w/np.sum(w))


class BoltzmannSampler(Sampler):
    def __init__(self, h, beta):
        h = h - h.max()
        w = np.exp(h * beta)
        w /= np.sum(w)

        self._w = w


def randomized_swap_optimization(c_0, candidates, cluster_h, sample_h,
                                 bound_function, evaluation_function,
                                 n_swaps, n_samples, logger=None):

    if logger:
        logger.info('Running random swapping routine')
        logger.info(f'n_clusters: {len(cluster_h)}, n_candidates: {len(sample_h)}')
        logger.info(f'n_swaps: {n_swaps}, n_samples:  {n_samples}')
        logger.weak_line()

    cluster_index_sampler_data = dict(sampling_class=PolynomialSampling, params=dict(exponent=1))
    cluster_center_sampler_data = dict(sampling_class=PolynomialSampling, params=dict(exponent=1))

    cluster_index_sampler = \
        cluster_index_sampler_data['sampling_class'](cluster_h, **cluster_index_sampler_data['params'])
    cluster_center_sampler = \
        cluster_center_sampler_data['sampling_class'](sample_h, **cluster_center_sampler_data['params'])

    max_v = evaluation_function(c_0)
    c_best = c_0

    for i in range(n_samples):
        c_i = c_0.copy()
        cluster_indexes = cluster_index_sampler.sample(n_swaps)
        candidate_indexes = cluster_center_sampler.sample(n_swaps)
        for cluster_index, candidate_index in zip(cluster_indexes, candidate_indexes):
            c_i[cluster_index] = candidates[candidate_index]

        if bound_function(c_i):
            v = evaluation_function(c_i)
            if v > max_v:
                max_v = v
                c_best = c_i

    return c_best
