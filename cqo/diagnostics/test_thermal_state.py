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
omega =2 * np.pi * 1.5e5
n = 0.01
beta = np.log(1 + 1/n) / (omega*hbar)

rho = ThermalState(beta, omega, mass)

limit = 1*rho.width

print(rho.sample(rho.width,rho.width))

N = 5

fidelity = np.zeros((N,))
normalisation = np.zeros((N,))

for i in range(1,N):

    x_array = np.linspace(-limit, limit, (2**i))

    width = x_array[1] - x_array[0]

    fidelity[i] = 2**i

    pdf = [rho.sample(x,x) for x in x_array]

    normalisation[i] = width*sum(pdf)


density_matrix = np.array([[rho.sample(x, x_) for x in x_array] for x_ in x_array])

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(x_array, x_array)

surf = ax.plot_surface(X, Y, np.abs(density_matrix), cmap=cm.coolwarm, linewidth=0,
                       antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('\\rho(x,x\')')

fig2 = plt.figure()
plt.plot(fidelity, normalisation)
plt.title('Norm vs resolution');
plt.show()
