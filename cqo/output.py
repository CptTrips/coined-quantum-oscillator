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
