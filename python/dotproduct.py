"""
  This program calculates the dot product of the wfc Bloch factor with their neighbors

"""

import sys
import time

import numpy as np

# This are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d


###################################################################################
def connection(nkconn, neighborconn, dphaseconn):
    """Calculates the dot product  of all combinations of wfc in nkconn and neighborconn."""

    dpc1 = np.zeros((d.nbnd, d.nbnd), dtype=complex)
    dpc2 = np.zeros((d.nbnd, d.nbnd), dtype=complex)

    for banda0 in range(d.nbnd):
        # reads first file for dot product
        infile = str(d.wfcdirectory) + "k0" + str(nkconn) + "b0" + str(banda0) + ".wfc"
        with open(infile, "rb") as fichconn:
            wfc0 = np.load(fichconn)
        fichconn.close()

        for banda1 in range(d.nbnd):
            # reads second file for dot product
            infile = (
                str(d.wfcdirectory)
                + "k0"
                + str(neighborconn)
                + "b0"
                + str(banda1)
                + ".wfc"
            )
            with open(infile, "rb") as fichconn:
                wfc1 = np.load(fichconn)
            fichconn.close()

            # calculates the dot products u_1.u_2* and u_2.u_1*
            dpc1[banda0, banda1] = np.sum(dphaseconn * wfc0 * np.conjugate(wfc1))
            dpc2[banda1, banda0] = np.conjugate(dpc1[banda0, banda1])

    return dpc1, dpc2


###################################################################################
if __name__ == "__main__":
    header("DOTPRODUCT", str(d.version), time.asctime())

    STARTTIME = time.time()  # Starts counting time

    # Reading data needed for the run

    WFCDIRECTORY = str(d.wfcdirectory)
    print("     Directory where the wfc are:", WFCDIRECTORY)
    NKS = d.nks
    print("     Total number of k-points:", NKS)

    NR = d.nr
    print("     Total number of points in real space:", NR)

    NPR = d.npr
    print("     Number of processors to use", NPR)

    NBND = d.nbnd
    print("     Number of bands:", NBND)
    print()

    PHASE = d.phase
    print("     Phases loaded")
    # print(PHASE[10000,10]) # PHASE[NR,NKS]

    neighbors = d.neighbors
    print("     Neighbors loaded")

    # Finished reading data needed for the run
    print()
    ##########################################################

    dpc = np.full((NKS, 4, NBND, NBND), 0 + 0j, dtype=complex)
    dp = np.zeros((NKS, 4, NBND, NBND))

    for nk in range(NKS):  # runs through all k-points
        for j in range(4):  # runs through all neighbors
            neighbor = neighbors[nk, j]

            if neighbor != -1 and neighbor > nk:  # exclude invalid neighbors
                jNeighbor = np.where(neighbors[neighbor] == nk)
                # Calculates the diference in phases to convert \psi to u
                dphase = PHASE[:, nk] * np.conjugate(PHASE[:, neighbor])

                print(
                    "      Calculating   nk = "
                    + str(nk)
                    + "  neighbor = "
                    + str(neighbor)
                )
                sys.stdout.flush()

                dpc[nk, j, :, :], dpc[neighbor, jNeighbor, :, :] = (
                    connection(nk, neighbor, dphase) / NR
                )

    dp = np.abs(dpc)

    # Save dot products to file
    with open("dpc.npy", "wb") as fich:
        np.save(fich, dpc)
    fich.close()
    print("     Dot products saved to file dpc.npy")

    # Save dot products modulus to file
    with open("dp.npy", "wb") as fich:
        np.save(fich, dp)
    fich.close()
    print("     Dot products modulus saved to file dp.npy")

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
