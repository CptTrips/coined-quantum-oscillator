from matplotlib import pyplot as plt
import numpy as np
from algebra import condition_subsystem


def output(state):

    # Create folder with time & date and key params

    # Output spatial probability distrubtions (up, down and total)
    state_up = condition_subsystem(state, np.array([1,0]))
    p_up = np.trace(state_up)
    state_up *= 1/p_up

    state_down = condition_subsystem(state, np.array([0,1]))
    p_down = np.trace(state_down)
    state_down *= 1/p_down

    reduced_state = p_up * state_up + p_down * state_down

    plt.figure()
    plt.plot(np.diag(state_up))
    plt.savefig("/home/matthewf/output_pdf_up.png")

    plt.figure()
    plt.plot(np.diag(state_down))
    plt.savefig("/home/matthewf/output_pdf_down.png")

    plt.plot(np.diag(reduced_state))
    plt.savefig("/home/matthewf/output_pdf.png")


    # Output graphical representation of final state
