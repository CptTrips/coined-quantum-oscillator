import numpy as np
from matplotlib import pyplot as plt
from cqo.units import hbar
from cqo.simulation import ThermalState

m = 1

omega = 1

t_array = np.linspace(0.75,1)

def A(t):

    A = np.array([
        [3/2, 1j, -t, -1j*t],
        [0, 1/2, 1j*t, -t],
        [0, 0, 3/2, -1j],
        [0, 0, 0, 1/2]])
    A = A + A.T

    return A

evals = np.array([np.linalg.eigvals(A(t)) for t in t_array])

plt.figure()

for i in range(4):
    plt.plot(t_array, evals[:,i])

print(evals[-1,:])

plt.show()

# Find t which makes A singular

t_min = 0.9

t_max = 1

min_e = 1

eps = 1e-15

while abs(min_e.real) > eps:

    t = (t_min + t_max) / 2

    min_e = np.linalg.eigvals(A(t)).min()

    if min_e.real > 0:
        t_min = t
    elif min_e.real < 0:
        t_max = t

    print("t_min: {}, t_max: {}, min_e: {}".format(t_min, t_max, min_e))
