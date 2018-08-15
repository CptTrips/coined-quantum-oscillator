import timeit
import numpy as np
from cqo.main import main as my_main
from cqo.simulation import (CoherentState,
                            expansion_protocol,
                            SpinState,
                            final_state)
from cqo.algebra import projector
from cqo.units import hbar
from matplotlib import pyplot as plt

def walk_params():

    N = 2

    mass = 3.5e3 * 4/3 * np.pi * (5e-8)**3

    omega = 1e5

    tscale = (2*np.pi)/omega

    lscale = np.sqrt(hbar / (2 * mass * omega))

    walk_state = CoherentState(0, mass*omega)

    spin_state = SpinState(projector(2,0))

    coin = (1/np.sqrt(2))*np.array([
        [1, 1j],
        [1j, 1]
    ])

    F = 1e-20
    F_enhancement = 1e2
    F = F * F_enhancement

    alpha = 2 * F / mass / omega**2

    Gamma_sc = 2 * np.pi * 1.1e2

    gamma = Gamma_sc / lscale**2

    T = tscale / 2

    gamma_T = gamma * T

    error = 5e-3

    return N, mass, omega, walk_state, spin_state, coin, alpha, gamma_T, error


def create_walk_mesh(res, width, N, alpha):

    sample_min = -2*width

    sample_max = N*alpha + 2*width

    sample_points = np.linspace(sample_min, sample_max, (N+1)*res)

    mesh_x, mesh_x_ = np.meshgrid(sample_points, sample_points)

    mesh = np.array([mesh_x, mesh_x_]).transpose(1, 2, 0)

    return mesh


def walk(res):

    N, mass, omega, walk_state, spin_state, coin, alpha, gamma_T, error = walk_params()

    mesh = create_walk_mesh(res, walk_state.width, N, alpha)


    element = lambda x, x_, s, s_: final_state(N, x, x_, s, s_, walk_state,
                                          spin_state, coin, alpha, gamma_T,
                                          error)

    rho_walk = np.array([
    [
        [
            [
                element(vec[0], vec[1], s, s_) for s_ in [0,1]
            ] for s in [0,1]
        ] for vec in row
    ] for row in mesh])


def create_mesh(res, width):

    coords = np.linspace(-3*width, 3*width, res)

    mesh_x, mesh_x_ = np.meshgrid(coords, coords)

    mesh = np.array([mesh_x, mesh_x_]).transpose(1, 2, 0)

    return mesh


def density_matrix(mass, omega, res):

    state = CoherentState(0, mass*omega)

    width = state.width

    mesh = create_mesh(res, width)

    matrix = np.array([[state.sample(x[0], x[1]) for x in row] for row in mesh])

    return matrix, mesh


def expansion(res):

    mass = 3.5e3 * 4/3 * np.pi * (5e-8)**3

    omega = 1e5

    t_free = 3/omega

    res_str = str(res)

    state, mesh = density_matrix(mass, omega, res)

    expansion_protocol(state, mesh, mass, omega, t_free)


def time_walk(res):

    statement = "walk("+str(res)+")"

    t = timeit.timeit(statement, number=1, globals=globals())

    print("Resolution: {}\nTime: {}".format(res, t))

    return t


def time_expansion(res):

    statement = "expansion("+str(res)+")"

    t = timeit.timeit(statement, number=1, globals=globals())

    print("Resolution: {}\nTime: {}".format(res, t))

    return t


def time_main(res):

    res_str = str(res)

    statement = "my_main(resolution="+res_str+")"

    print(statement)

    return timeit.timeit(statement, number=1, globals=globals())


def benchmark(timer, limit):

    resolutions = 2**np.arange(1,limit)

    timings = [timer(res) for res in resolutions]

    plt.figure()
    plt.plot(np.log(resolutions), np.log(timings))
    plt.show(block=False)



if __name__ == "__main__":

    print("Timing walk")

    benchmark(time_walk, 6)

    print("Timing expansion")

    benchmark(time_expansion, 7)
