import numpy as np
b = .5
a = .5

e_a = np.array([0., 1, 1, 3])
e_b = np.array([0., 1., 0., 6.])

p_a = np.array([(1-b) * (1-b), (1-b) * b, b * (1-a), b * a])
p_b = np.array([(1 - a * b) * (1-b), (1 - a * b) * b, (a * b) * (1 - a * b), (a * b) ** 2])

print('p_a', p_a)
print('p_b', p_b)
assert (np.abs(np.sum(p_a) - 1.) < 1e-16)
assert (np.abs(np.sum(p_b) - 1.) < 1e-16)


print('exp a', np.sum(e_a * p_a))
print('exp b', np.sum(e_b * p_b))

