from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from itertools import product
from cqo.simulation import ThermalState
from cqo.units import hbar
import numpy as np

# Display a thermal density matrix and show it is normalised to 1

output_folder = '/home/matthewf/PhD/programming/coined-quantum-oscillator/output'

mass = 3.51e3 * 4/3 * np.pi * (5e-8)**3
omega = 2 * np.pi * 1.5e5
n = 10
beta = np.log(1 + 1/n) / (omega*hbar)

rho = ThermalState(beta, omega, mass)

print(rho.sample(0,0) / rho.sample(rho.width, rho.width))

limit = 3*rho.width

N = 4

fidelity = np.zeros((N,))
normalisation = np.zeros((N,))

for i in range(1,N):

    res = 16 * 2**i

    x_array = np.linspace(-limit, limit, res)

    width = x_array[1] - x_array[0]

    fidelity[i] = 2**i

    pdf = [rho.sample(x,x) for x in x_array]

    normalisation[i] = width*sum(pdf)

    density_matrix = np.array([[rho.sample(x, x_) for x in x_array] for x_ in x_array])

    X, Y = np.meshgrid(x_array, x_array)

    fig = plt.figure()

    plt.subplot(2,1,1)
    plt.pcolor(X, Y, np.abs(density_matrix))
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.plot(np.diag(X), np.diag(density_matrix))

    plt.title('\\rho(x,x\')')


fig2 = plt.figure()
plt.plot(fidelity, normalisation)
plt.ylim(ymin=0)
plt.title('Norm vs resolution');

plt.show()
