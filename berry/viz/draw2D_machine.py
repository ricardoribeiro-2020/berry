"""
# This program draws the bands
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from berry._subroutines.contatempo import tempo
from berry._subroutines.headerfooter import header, footer
import berry._subroutines.loaddata as d

# pylint: disable=C0103
###################################################################################
def main():
    header("DRAWBANDS", d.version, time.asctime())

    starttime = time.time()  # Starts counting time

    if len(sys.argv) != 3:
        print(" ERROR in number of arguments. Has to have two integers.")
        print(" The first is the first band, the second is last band.")
        sys.exit("Stop")

    startband = 0  # Number of the first band
    endband = 7  # Number of the last band

    fig = plt.figure(figsize=(6, 6))

    cores = [
        "black",
        "blue",
        "green",
        "red",
        "grey",
        "brown",
        "violet",
        "seagreen",
        "dimgray",
        "darkorange",
        "royalblue",
        "darkviolet",
        "maroon",
        "yellowgreen",
        "peru",
        "steelblue",
        "crimson",
        "silver",
        "magenta",
        "yellow",
    ]

    # Reading data needed for the run
    berrypath = str(d.berrypath)
    print(" Path to BERRY files:", berrypath)

    wfcdirectory = str(d.wfcdirectory)
    print(" Directory where the wfc are:", wfcdirectory)
    print(" Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print(" Total number of k-points:", d.nks)
    print(" Number of bands:", d.nbnd)
    print()
    print(" Eigenvlaues loaded")
    print(" K-points loaded")


    xarray = np.zeros((d.nkx, d.nky))
    yarray = np.zeros((d.nkx, d.nky))
    zarray = np.zeros((d.nkx, d.nky))
    count = -1
    for j in range(d.nky):
        for i in range(d.nkx):
            count = count + 1
            xarray[i, j] = d.kpoints[count, 0]
            yarray[i, j] = d.kpoints[count, 1]

    ax = fig.gca()
    for banda in range(startband, endband + 1):
        count = -1
        for j in range(d.nky):
            for i in range(d.nkx):
                count = count + 1
                zarray[i, j] = d.eigenvalues[count, banda]

        ax.plot(xarray, yarray, zarray, color=cores[banda])

    # Para desenhar no mathematica!
    #
    # print('b'+str(banda)+'={', end = '')
    # for beta in range(d.nky):
    #   print('{', end = '')
    #   for alfa in range(d.nkx):
    #     if alfa != d.nkx-1:
    #       print(str(zarray[alfa][beta])+str(','), end = '')
    #     else:
    #       print(str(zarray[alfa][beta]), end = '')
    #   if beta != d.nky-1:
    #     print('},')
    #   else:
    #     print('}', end = '')
    # print('};\n')


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_trisurf(xarray, yarray, zarray, linewidth=0.2, antialiased=True)

    plt.show()


    #    sys.exit("Stop")

    # Finished
    endtime = time.time()

    footer(tempo(starttime, endtime))

if __name__ == "__main__":
    main()
