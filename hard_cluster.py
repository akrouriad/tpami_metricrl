import numpy as np
import torch
from torch.autograd import Function


class HardClusterMembership(Function):
    """
    Take as input a set of cluster scores, a set of cluster weights, and returns the (hard) cluster membership vector,
    that is a vector with the value of 1 at the index of the cluster with higher score*weight and 0 elsewhere  .
    """
    @staticmethod
    def forward(ctx, input, weights):
        values = input.mm(torch.diag(weights))
        max_values = values.argmax(1)
        ctx.save_for_backward(input, weights, values, max_values)
        clusters = torch.zeros(input.size()).scatter_(1, max_values.unsqueeze(1), 1)

        return clusters

    @staticmethod
    def backward(ctx, grad_output):
        """
        This function computes the approximate gradiend by considering the minimum epsilon for which we change the
        cluster.
        :param ctx:
        :param grad_output:
        :return:
        """
        input, weights, values, max_values = ctx.saved_tensors

        n_clusters = input.shape[1]

        grad = torch.zero([n_clusters, n_clusters])

        for c in range(n_clusters):
            eps_min = 0
            eps_max = 0

            for i, x in enumerate(input):
                if max_values[i] == c:
                    v2 = torch.topk(values[i], 2).values[1]
                    delta = weights[c] - v2 / x[c]
                    eps_min = np.min(eps_min, delta)
                else:
                    v2 = torch.max(values[i])
                    delta = v2 / x[c] - weights[c]
                    eps_max = np.min(eps_max, delta)

            


        return grad.mm(grad_output)
