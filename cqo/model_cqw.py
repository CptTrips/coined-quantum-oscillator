from cqo import simulation, output
import numpy as np


def run_model():

    N = 300

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    coin_op = (1 / np.sqrt(2)) * sigma_x @ (sigma_x + sigma_z)

    amplitudes = simulation.walk_amplitudes(N, coin_op, initial_state = [0,1])

    probabilities = np.abs(amplitudes)**2

    x = range(len(probabilities[:,0]))

    output.draw_pdf(x, probabilities[:,0], probabilities[:,1], "CQW Model 0")

    return probabilities
