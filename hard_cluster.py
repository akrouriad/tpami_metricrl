import torch
from torch.autograd import Function


class HardClusterMembership(Function):
    """
    Take as input a set of cluster scores, a set of cluster weights, and returns the (hard) cluster membership vector,
    that is a vector with the value of 1 at the index of the cluster with higher score*weight and 0 elsewhere  .
    """
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        values = input.mm(torch.diag(weights))
        print('values', values)
        max_values = values.argmax(1)
        print('max_values_indices', max_values)
        clusters = torch.zeros(input.size()).scatter_(1, max_values.unsqueeze(1), 1)
        print('clusters', max_values)
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
        pass
