import numpy as np
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter


class HardClusterMembership(Function):
    """
    Take as input a set of cluster scores, a set of cluster weights, and returns the (hard) cluster membership vector,
    that is a vector with the value of 1 at the index of the cluster with higher score*weight and 0 elsewhere  .
    """

    @staticmethod
    def eval_all(inputs, weights):
        values = inputs.mm(torch.diag(weights))
        max_values = values.argmax(1)
        clusters = torch.zeros(inputs.size()).scatter_(1, max_values.unsqueeze(1), 1)

        return values, max_values, clusters

    @staticmethod
    def eval_clusters(inputs, weights):
        _, _, clusters = HardClusterMembership.eval_all(inputs, weights)

        return clusters

    @staticmethod
    def forward(ctx, inputs, weights):
        values, max_values, clusters = HardClusterMembership.eval_all(inputs, weights)
        ctx.save_for_backward(inputs, weights, values, max_values)

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
        inputs, weights, values, max_values = ctx.saved_tensors

        n_points = inputs.shape[0]
        n_clusters = inputs.shape[1]

        grad = torch.zeros([n_points, n_clusters, n_clusters])

        eps_min_list = list()
        eps_max_list = list()

        for c in range(n_clusters):
            eps_min = np.infty
            eps_max = np.infty

            for i, x in enumerate(inputs):
                if max_values[i] == c:
                    v2 = torch.topk(values[i], 2)[0][1]
                    delta = weights[c] - v2 / x[c]
                    eps_min = np.minimum(eps_min, delta.item())
                else:
                    v2 = torch.max(values[i])
                    delta = v2 / x[c] - weights[c]
                    eps_max = np.minimum(eps_max, delta.item())

            # eps_min = 1 if np.isinf(eps_min) else eps_min
            # eps_max = 1 if np.isinf(eps_max) else eps_max

            eps_min_list.append(eps_min)
            eps_max_list.append(eps_max)

        for c in range(n_clusters):
            eps_min = eps_min_list[c]
            eps_max = eps_max_list[c]

            new_weights_plus = weights.clone()
            new_weights_plus[c] += eps_max

            new_weights_minus = weights.clone()
            new_weights_minus[c] -= eps_min

            print('step for cluster c:', eps_max, eps_min)

            grad[:, c, :] = (HardClusterMembership.eval_clusters(inputs, new_weights_plus) -
                             HardClusterMembership.eval_clusters(inputs, new_weights_minus)) / (eps_max + eps_min)

        weights_grad = torch.zeros([n_points, n_clusters])

        for i, g_output_x in enumerate(grad_output):
            # print(grad[i])
            # print(g_output_x)

            weights_grad[i] = grad[i].t().mv(g_output_x)

        return None, weights_grad


if __name__ == '__main__':
    n_steps = 1000
    n_cluster = 3
    n_points = 10

    w = torch.rand(n_cluster, dtype=torch.float, requires_grad=True)

    cost_w = torch.Tensor([0.8, 0.15, 0.05])

    phi = HardClusterMembership.apply
    x = torch.rand(n_points, n_cluster, dtype=torch.float)

    for t in range(n_steps):
        clusters = phi(x, w**2)
        loss = torch.norm(clusters.mv(cost_w))
        loss.backward()

        print('w', w.detach().numpy())
        print('input:')
        print(x.detach().numpy())
        print('clusters:')
        print(clusters.detach().numpy())
        print('loss:', loss.item())
        print('grad', w.grad.detach().numpy())

        print ('-----------------------------------------------------------------------------------------------------')


        with torch.no_grad():
            w += 0.01 * w.grad
            w.grad.zero_()

        input()