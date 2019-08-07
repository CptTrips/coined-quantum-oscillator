from cqo import simulation, output, algebra
import numpy as np


def run_model(N=100):

    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    coin_op = (1 / np.sqrt(2)) * (sigma_y - sigma_z)

    amplitudes_0 = simulation.walk_amplitudes(N, coin_op, initial_state = [1,0])

    amplitudes_1 = simulation.walk_amplitudes(N, coin_op, initial_state = [0,1])

    probabilities = 1*np.abs(amplitudes_0)**2 + 0*np.abs(amplitudes_1)**2

    binomial_probabilities = np.array(algebra.binomial_distribution(N))
    binomial_probabilities /= sum(binomial_probabilities)

    x = range(len(probabilities[:,0]))

    title_string = "CQW ({} steps)".format(N)

    output.draw_walk(x, probabilities[:,0], probabilities[:,1], binomial_probabilities, title_string)

    return probabilities


if __name__ == "__main__":

    run_model()
