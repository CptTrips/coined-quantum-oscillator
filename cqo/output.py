from matplotlib import pyplot as plt
import numpy as np
from algebra import condition_subsystem


def output(state, P_x):

    #! Create folder with time & date and key params

    # Plot site probability distrubtions (up, down and total)
    state_up, p_up = condition_subsystem(state, np.array([1,0]))
    state_down, p_down = condition_subsystem(state, np.array([0,1]))

    reduced_state = p_up * state_up + p_down * state_down

    #! Put these in one figure
    plt.figure()
    plt.plot(np.diag(state_up))
    plt.savefig("/home/matthewf/output_pdf_up.png")

    plt.figure()
    plt.plot(np.diag(state_down))
    plt.savefig("/home/matthewf/output_pdf_down.png")

    plt.plot(np.diag(reduced_state))
    plt.savefig("/home/matthewf/output_pdf.png")


    #! Output graphical representation of final state

    #! Plot spatial probability distribution
    plt.figure()
    plt.plot(P_x)
    plt.savefig("/home/matthewf/spatial_pdf.png")
