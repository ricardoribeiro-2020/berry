"""
 This program reads the dot products and eigenvalues and establishes the bands
"""

import sys
import time

from random import randrange
import math
import itertools
import numpy as np

import contatempo
from headerfooter import header, footer
import loaddata as d

###################################################################################
if __name__ == "__main__":
    header("COMPARA", time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) > 1:  # To enter an initial value for k-point (optional)
        firskpoint = int(sys.argv[1])
    else:
        firskpoint = -1

    WFCDIRECTORY = str(d.wfcdirectory)
    print("     Directory where the wfc are:", WFCDIRECTORY)
    NKX = d.nkx
    NKY = d.nky
    NKZ = d.nkz
    print("     Number of k-points in each direction:", NKX, NKY, NKZ)
    NKS = d.nks
    print("     Total number of k-points:", NKS)

    NBND = d.nbnd
    print("     Number of bands:", NBND)
    print()

    neighbors = d.neighbors
    print("     Neighbors loaded")

    eigenvalues = d.eigenvalues
    print("     Eigenvalues loaded")
    # print(eigenvalues)

    connections = np.load("dp.npy")
    print("     Modulus of direct product loaded")

    print()
    print("     Finished reading data")
    print()
    ##########################################################################
    bands = np.full((NKS, NBND, 100), -1, dtype=int)
    for bnd in range(NBND):
        bands[:, bnd, 0] = bnd  # Initial attribution of band numbers

    ##########################################################################
    TOL = 0.9
    NTENTATIVE = 5  # Nr of tentatives (doesn't seem useful in this configuration
    INITIALKS = []  # List to save the initial k-points that are choosen randomly
    # = nr of matches, -1 if contradictory matches
    signal = np.zeros((NKS, NBND, NTENTATIVE + 1), dtype=int)
    # Create arrays of tentatives
    for tentative in range(NTENTATIVE):
        if 0 <= firskpoint < NKS and tentative == 0:  # If initial k-point is given
            kp0 = firskpoint
        else:
            kp0 = randrange(NKS)  # Chooses a k-point randomly
        print("     starting k-point:", kp0)
        INITIALKS.append(kp0)  # Stores the departure k-point for future reference
        listdone = [kp0]  # List to store the k-points that have been analysed
        listk = []
        # initializes 1st k-point of the series
        bands[kp0, :, tentative + 1] = bands[kp0, :, 0]
        signal[kp0, :, 1] = 1  # First k-point has signal 1

        for i in range(4):  # Calculates the four points around the first
            for b1, b2 in itertools.product(range(NBND), range(NBND)):
                # Finds connections between k-points/bands
                if connections[kp0, i, b1, b2] > TOL:
                    bands[neighbors[kp0, i], b2, tentative + 1] = bands[
                        kp0, b1, tentative + 1
                    ]
                    signal[neighbors[kp0, i], b2, 1] += 1  # Signal a connection
            if (
                neighbors[kp0, i] not in listk
                and neighbors[kp0, i] not in listdone
                and neighbors[kp0, i] != -1
            ):
                # Adds neighbors not already done, for the next loop
                listk.append(neighbors[kp0, i])

        while len(listk) > 0:  # Runs through the list of neighbors not already done
            nk = listk[0]  # Chooses the first from the list
            for i in range(4):  # Calculates the four points around
                for b1, b2 in itertools.product(range(NBND), range(NBND)):
                    # Finds connections between k-points/bands
                    if connections[nk, i, b1, b2] > TOL:
                        # If that band is not valid, cycles the loop
                        if (
                            bands[nk, b1, tentative + 1] == -1
                            or signal[neighbors[nk, i], b2, tentative + 1] == -1
                        ):
                            break
                        # If the new band is not attributted, attribute it
                        if bands[neighbors[nk, i], b2, tentative + 1] == -1:
                            bands[neighbors[nk, i], b2, tentative + 1] = bands[
                                nk, b1, tentative + 1
                            ]
                            signal[
                                neighbors[nk, i], b2, tentative + 1
                            ] += 1  # Signal a connection
                        elif (
                            bands[neighbors[nk, i], b2, tentative + 1]
                            == bands[nk, b1, tentative + 1]
                        ):
                            signal[
                                neighbors[nk, i], b2, tentative + 1
                            ] += 1  # Signal a connection
                        else:
                            signal[
                                neighbors[nk, i], b2, tentative + 1
                            ] = -1  # Signal a contradiction

                if (
                    neighbors[nk, i] not in listk
                    and neighbors[nk, i] not in listdone
                    and neighbors[nk, i] != -1
                ):
                    listk.append(neighbors[nk, i])
            ##           print(nk,i,bands[neighbors[nk,i],:,1])
            listk.remove(nk)  # Remove k-point from the list of todo
            listdone.append(nk)  # Add k-point to the list of done

    listdone.sort()
    # print(listdone)

    ##########################################################################
    print()
    print("     Correcting problems found")
    print()
    ##########################################################

    negcount = 0
    for c in range(1):
        # Select points signaled -1
        kpproblem, bnproblem = np.where(signal[:, :, 1] == -1)

        # Create array with (kp,b1,b2) with the two problematic bands of each k-point
        # in array kpproblem, the k-points are in pairs, so make use of it
        problemlength = int(kpproblem.size / 2)
        kpb1b2 = np.zeros((problemlength, 3), dtype=int)

        for i in range(problemlength):
            kpb1b2[i, 0] = kpproblem[i * 2]
            kpb1b2[i, 1] = bnproblem[i * 2]
            kpb1b2[i, 2] = bnproblem[i * 2 + 1]

            print(
                "     Neighbors of the k-point with problem: ",
                kpb1b2[i, 0],
                neighbors[kpb1b2[i, 0], :],
            )
            print(bands[kpb1b2[i, 0], :, 1])
            validneig = np.count_nonzero(neighbors[kpb1b2[i, 0], :] != -1)
            count11 = count12 = count21 = count22 = 0
            for neig in range(4):
                if neighbors[kpb1b2[i, 0], neig] != -1:
                    print(bands[neighbors[kpb1b2[i, 0], neig], :, 1])
                    print(
                        kpb1b2[i, 0],
                        neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2],
                        bnproblem[i * 2],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2],
                        bnproblem[i * 2 + 1],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2 + 1]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2 + 1],
                        bnproblem[i * 2],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2 + 1], bnproblem[i * 2]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2 + 1],
                        bnproblem[i * 2 + 1],
                        connections[
                            kpb1b2[i, 0],
                            neig,
                            bnproblem[i * 2 + 1],
                            bnproblem[i * 2 + 1],
                        ],
                    )

                    if (
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2]
                        ]
                        > 0.85
                    ):
                        count11 += 1
                    if (
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2 + 1]
                        ]
                        > 0.85
                    ):
                        count12 += 1
                    if (
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2 + 1], bnproblem[i * 2]
                        ]
                        > 0.85
                    ):
                        count21 += 1
                    if (
                        connections[
                            kpb1b2[i, 0],
                            neig,
                            bnproblem[i * 2 + 1],
                            bnproblem[i * 2 + 1],
                        ]
                        > 0.85
                    ):
                        count22 += 1

            # See if problem can be solved:
            if count11 == validneig:
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2], 1
                ] = validneig  # signals problem as solved
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2 + 1], 1
                ] = validneig  # signals problem as solved
                print("     Solved.")
                negcount += 1
            elif count12 == validneig:
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2], 1
                ] = validneig  # signals problem as solved
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2 + 1], 1
                ] = validneig  # signals problem as solved
                print("     Solved.")
                negcount += 1
            elif count21 == validneig:
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2], 1
                ] = validneig  # signals problem as solved
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2 + 1], 1
                ] = validneig  # signals problem as solved
                print("     Solved.")
                negcount += 1
            elif count22 == validneig:
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2], 1
                ] = validneig  # signals problem as solved
                signal[
                    kpb1b2[i, 0], bnproblem[i * 2 + 1], 1
                ] = validneig  # signals problem as solved
                print("     Solved.")
                negcount += 1

    ##########################################################################
    # Try attribution through eigenvalue continuity
    eigcont = 0
    for c, nk0, nb0 in itertools.product(range(3), range(NKS), range(NBND)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = neighbors[nk0, neig]
                for nb1 in range(NBND):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.01
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(NKS), range(NBND)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = neighbors[nk0, neig]
                for nb1 in range(NBND):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.01
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.02
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(NKS), range(NBND)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = neighbors[nk0, neig]
                for nb1 in range(NBND):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.01
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.02
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.03
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(NKS), range(NBND)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = neighbors[nk0, neig]
                for nb1 in range(NBND):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.01
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.02
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.04
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            eigenvalues[nk0, nb0], eigenvalues[nk1, nb1], abs_tol=0.08
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    ##########################################################################
    print()
    print("     Try a more relaxed attribution where it failed")
    print()
    attcount = 0
    # Select points signaled 0
    kpproblem, bnproblem = np.where(signal[:, :, 1] == 0)
    problemlength = int(kpproblem.size)
    kpb1b2 = np.zeros((problemlength, 2), dtype=int)
    for i in range(problemlength):
        kpb1b2[i, 0] = kpproblem[i]
        kpb1b2[i, 1] = bnproblem[i]
        validneig = np.count_nonzero(neighbors[kpb1b2[i, 0], :] != -1)
        count11 = 0
        refbnd = -1
        for neig in range(4):
            if neighbors[kpb1b2[i, 0], neig] != -1:
                for j in range(NBND):
                    if (
                        connections[
                            kpb1b2[i, 0],
                            neig,
                            bnproblem[i],
                            bands[neighbors[kpb1b2[i, 0], neig], j, 1],
                        ]
                        > 0.8
                        and bands[neighbors[kpb1b2[i, 0], neig], j, 1] != -1
                    ):
                        print(
                            kpb1b2[i, 0],
                            neighbors[kpb1b2[i, 0], neig],
                            bnproblem[i],
                            bands[neighbors[kpb1b2[i, 0], neig], j, 1],
                            connections[
                                kpb1b2[i, 0],
                                neig,
                                bnproblem[i],
                                bands[neighbors[kpb1b2[i, 0], neig], j, 1],
                            ],
                        )
                        if refbnd == -1:
                            refbnd = bands[neighbors[kpb1b2[i, 0], neig], j, 1]
                            count11 += 1
                        elif refbnd == bands[neighbors[kpb1b2[i, 0], neig], j, 1]:
                            count11 += 1
                        else:
                            count11 = -100
        if count11 > 0:
            print("     Found band to attribute!")
            bands[kpb1b2[i, 0], kpb1b2[i, 1], 1] = refbnd
            signal[kpb1b2[i, 0], kpb1b2[i, 1], 1] = count11
            attcount += 1

    ##########################################################################
    bandsfinal = np.full((NKS, NBND), -1, dtype=int)  # Array for the final results
    # gives the machine band that belongs to band (nk,nb)
    signalfinal = np.zeros((NKS, NBND), dtype=int)  # Array for final signalling
    first = True
    attrib = []
    merger = 1
    cont = 1
    for i in INITIALKS:  # Runs through all sets of bands
        if first:  # First set is special
            first = False
            bandsfinal = bands[:, :, 1]  # Starts with the first set
            signalfinal = signal[:, :, 1].astype(int)  # Starts with first set signaling
            for nk in range(NKS):
                if np.all(bandsfinal[nk, :] != -1):
                    attrib.append(
                        nk
                    )  # finds kpoints with all bands attributed in the first set
            continue
        cont += 1
        for j in attrib:  # Runs through all kpoints with all bands attributed
            # if the new set has also at the same kpoint all bands attributed, merge them
            if np.all(bands[j, :, cont] != -1):
                if np.all(bands[j, :, 1] == bands[j, :, cont]):
                    ##               print(j,bands[j,:,cont],bands[j, :, 1])
                    merger += 1
                    print("     ***** Found same order! Merge.")
                    for nk in range(NKS):
                        for nb in range(NBND):
                            if (
                                bandsfinal[nk, nb] == bands[nk, nb, cont]
                                or bands[nk, nb, cont] == -1
                            ):
                                continue
                            if bandsfinal[nk, nb] == -1 and bands[nk, nb, cont] != -1:
                                bandsfinal[nk, nb] = bands[nk, nb, cont]
                                signalfinal[nk, nb] = 1
                                print("     Changed ", nk, nb)
                            elif signalfinal[nk, nb] == -1:
                                continue
                            else:
                                signalfinal[nk, nb] = -2
                                print(nk, nb, bandsfinal[nk, nb], bands[nk, nb, cont])
                                print("     !! Found incompatibility")

                    break

    # sys.exit("Stop")

    ###################################################################################
    print()
    print("     *** Final Report ***")
    print()
    nrnotattrib = np.full((NBND), -1, dtype=int)
    SEP = " "
    print("     Bands: gives the machine nr that belongs to new band (nk,nb)")
    for nb in range(NBND):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(bandsfinal[:, nb] == -1)
        print()
        print(
            "  New band "
            + str(nb)
            + "      | y  x ->   nr of fails: "
            + str(nrnotattrib[nb])
        )
        for j in range(NKY):
            lin = ""
            print()
            for i in range(NKX):
                nk = nk + 1
                f = bandsfinal[nk, nb]
                if f < 0:
                    lin += SEP + str(f)
                elif 0 <= f < 10:
                    lin += SEP + SEP + str(f)
                elif 9 < f < 100:
                    lin += SEP + str(f)
            print(lin)
    print()
    print(" Signaling: how many events in each band signaled.")
    nrsignal = np.full((NBND, 7), -2, dtype=int)
    for nb in range(NBND):
        nk = -1
        nrsignal[nb, 0] = str(np.count_nonzero(signalfinal[:, nb] == -1))
        nrsignal[nb, 1] = str(np.count_nonzero(signalfinal[:, nb] == 0))
        nrsignal[nb, 2] = str(np.count_nonzero(signalfinal[:, nb] == 1))
        nrsignal[nb, 3] = str(np.count_nonzero(signalfinal[:, nb] == 2))
        nrsignal[nb, 4] = str(np.count_nonzero(signalfinal[:, nb] == 3))
        nrsignal[nb, 5] = str(np.count_nonzero(signalfinal[:, nb] == 4))
        nrsignal[nb, 6] = str(np.count_nonzero(signalfinal[:, nb] == 5))
        print()
        print(
            "     "
            + str(nb)
            + "      | y  x ->   -1: "
            + str(nrsignal[nb, 0])
            + "     0: "
            + str(nrsignal[nb, 1])
        )
        for j in range(NKY):
            lin = ""
            print()
            for i in range(NKX):
                nk = nk + 1
                f = signalfinal[nk, nb]
                if f < 0:
                    lin += SEP + str(f)
                elif 0 <= f < 10:
                    lin += SEP + SEP + str(f)
                elif 9 < nk < 100:
                    lin += SEP + str(f)
            print(lin)

    print()
    print(" Resume of results")
    print()
    print(" nr k-points not attributed to a band (bandfinal=-1)")
    print(" Band       nr k-points")
    for nb in range(NBND):
        print(" ", nb, "         ", nrnotattrib[nb])

    print()
    print(" Signaling: how many events in each band signaled.")
    print(" Band   -1   0   1   2   3   4   5")
    for nb in range(NBND):
        print("  " + str(nb) + "   " + str(nrsignal[nb, :]))

    print()
    print("     Continuity recovered: ", negcount)
    print("     Found by eigenvalue continuity: ", eigcont)
    print("     More relaxed attribution: ", attcount)
    print("     Merged ", merger, " sets")
    print()

    print("     Saving files bandsfinal.npy and signalfinal.npy")
    print("     (bandsfinal.npy gives the machine number for each k-point/band)")
    with open("bandsfinal.npy", "wb") as f:
        np.save(f, bandsfinal)
    f.close()
    with open("signalfinal.npy", "wb") as f:
        np.save(f, signalfinal)
    f.close()

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
