from matplotlib import pyplot as plt
import numpy as np
from cqo.algebra import condition_subsystem


def output(x, P_walk, x_final, P_final):

    #! Create folder with time & date and key params

    #! Output graphical representation of final state

    plt.figure()
    plt.plot(x, P_walk)
    plt.axis(ymin=0)
    plt.title("Walk pdf")

    plt.figure()
    plt.plot(x_final, P_final)
    plt.axis(ymin=0)
    plt.title("Final pdf")

    plt.show()
    #plt.savefig("/home/matthewf/spatial_pdf.png")


def draw_pdf(x, P, title, block_arg=True):

    plt.figure()

    plt.plot(x, P)

    plt.axis(ymin=0)

    plt.title(title)

    plt.show(block=block_arg)


def draw_state(x, P_0, P_1, title, show=True):

    plt.figure()

    plt.plot(x, P_0, label = "P_0")

    plt.plot(x, P_1, label = "P_1")

    plt.plot(x, P_0 + P_1, label = "P_tot")

    plt.legend()

    plt.title(title)

    if show:
        plt.show()
