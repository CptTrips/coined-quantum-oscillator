import timeit
import numpy as np
from cqo import main
from matplotlib import pyplot as plt


def time_main(res):

    res_str = str(res)

    statement = "main(resolution="+res_str+")"

    return timeit.timeit(statement, number=1)


def benchmark_main():

    resolutions = [4, 8, 16, 32]

    timings = [time_main(res) for res in resolutions]

    plt.figure()
    plt.plot(resolutions, timings)
    plt.show()



if __name__ == "__main__":
    benchmark_main()
