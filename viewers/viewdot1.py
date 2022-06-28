"""
 This program reads the dot products and prints them
"""
import sys
import numpy as np
from colorama import Fore, Back, Style
import loaddata as d

###################################################################################
if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("     ERROR in number of arguments. Needs the band number.")

    band1 = int(sys.argv[1])

    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)
    print("     Band calculated:", band1)

    connections = np.load("dp.npy")
    valuesarray = np.zeros((d.nks, d.nbnd), dtype=float)
    bandsarray = np.zeros((d.nks, d.nbnd), dtype=int)
    #    print(connections.shape)

    coresback = [
        Back.BLACK,
        Back.BLUE,
        Back.GREEN,
        Back.RED,
        Back.YELLOW,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
        Back.BLACK,
        Back.RED,
        Back.GREEN,
        Back.YELLOW,
        Back.BLUE,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
        Back.BLACK,
        Back.RED,
        Back.GREEN,
        Back.YELLOW,
        Back.BLUE,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
    ]
    coresfore = [
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
    ]

    for j in range(4):
        for i in range(d.nks):
            nk = d.neighbors[i, j]
            for band in range(d.nbnd):
                bandsarray[i, band] = np.argmax(connections[i, j, band, :])
                valuesarray[i, band] = np.amax(connections[i, j, band, :])

        nk = -1
        SEP = " "
        precision = 1
        print(
            "         | y  x ->  "
            + coresback[0]
            + coresfore[0]
            + " 0 "
            + coresback[1]
            + coresfore[1]
            + " 1 "
            + coresback[2]
            + coresfore[2]
            + " 2 "
            + coresback[3]
            + coresfore[3]
            + " 3 "
            + coresback[4]
            + coresfore[4]
            + " 4 "
            + coresback[5]
            + coresfore[5]
            + " 5 "
            + coresback[6]
            + coresfore[6]
            + " 6 "
            + coresback[7]
            + coresfore[7]
            + " 7 "
            + Style.RESET_ALL
        )
        for _ in range(d.nky):
            lin = ""
            print()
            for _ in range(d.nkx):
                nk = nk + 1
                val = (
                    coresback[bandsarray[nk, band1]]
                    + coresfore[bandsarray[nk, band1]]
                    + "{0:.{1}f}".format(valuesarray[nk, band1], precision)
                    + Style.RESET_ALL
                )
                if valuesarray[nk, band1] < 0:
                    lin += SEP + str(val)
                elif 0 <= valuesarray[nk, band1] < 10:
                    lin += SEP + SEP + str(val)
                elif 9 < valuesarray[nk, band1] < 100:
                    lin += SEP + str(val)
            print(lin)

        print()
        print()
