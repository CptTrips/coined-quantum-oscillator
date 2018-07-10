from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from simulation import final_state, final_state_recursive
from simulation import CoherentState, SpinState
from algebra import projector
import timeit

N = 5

mass = 1
omega = 1
alpha_0 = 0
alpha_array = [5]
gamma_array = [5e-2, 1e-1]

resolution = 7
error = 1e-6

Haddamard = (1/np.sqrt(2))*np.array([
    [1, 1],
    [1, -1]
])

balanced_flip =(1/np.sqrt(2))*np.array([
    [1, 1j],
    [1j, 1]
])

coin_array = [balanced_flip]

walk_state = CoherentState(alpha_0, mass*omega)

spin_state = SpinState(projector(2,0))

def integrate(P_x, width):
    """
    Calculate the area under a curve using a trapezium approximation.
    """

    area = 0

    for i in range(len(P_x)-1):

        area += P_x[i] * width + 0.5 * width * (P_x[i+1] - P_x[i])

    return area


for alpha, gamma, coin in product(alpha_array, gamma_array, coin_array):

    sample_points = np.linspace(-alpha, (N+1)*alpha, (N+3)*resolution)

    width = sample_points[1] - sample_points[0]

    pdf_iterative = np.array([(final_state(N, x, x, 0, 0, walk_state,
                              spin_state, coin, alpha, gamma, error)
                  + final_state(N, x, x, 1, 1, walk_state,
                                spin_state, coin, alpha, gamma, error))
                 for x in sample_points])

    gamma = gamma / 1

    pdf_recursive = np.array([(final_state_recursive(N, x, x, 0, 0, walk_state,
                              spin_state, coin, alpha, gamma, error)
                  + final_state_recursive(N, x, x, 1, 1, walk_state,
                                spin_state, coin, alpha, gamma, error))
                 for x in sample_points])


    plt.figure()

    plt.subplot(3,1,1)
    plt.plot(sample_points, pdf_iterative)
    plt.title('Iterative (P = {})'.format(integrate(pdf_iterative, width)))

    plt.subplot(3,1,2)
    plt.plot(sample_points, pdf_recursive)
    plt.title('Recursive (P = {})'.format(integrate(pdf_recursive, width)))

    plt.subplot(3,1,3)
    plt.plot(sample_points, pdf_recursive - pdf_iterative)
    plt.title('R - I')

    plt.suptitle('alpha = {}, gamma = {}'.format(alpha, gamma))

plt.show()

print('Timing methods')

def time_iterative():
    final_state(5, 10, 10, 0, 0, walk_state, spin_state, balanced_flip, 2, 1e-3, 1e-4)

def time_recursive():
    final_state_recursive(5, 10, 10, 0, 0, walk_state, spin_state, balanced_flip, 2, 1e-3, 1e-4)

t_i = timeit.timeit('time_iterative()', setup='from __main__ import time_iterative',
                    number=100)
print("Iterative time: {}".format(t_i))

t_r = timeit.timeit('time_recursive()', setup='from __main__ import time_recursive',
                    number=100)
print("Recursive time: {}".format(t_r))
