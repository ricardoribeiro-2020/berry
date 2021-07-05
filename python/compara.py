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
from write_k_points import bands_numbers

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("COMPARA", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) > 1:  # To enter an initial value for k-point (optional)
        firskpoint = int(sys.argv[1])
    else:
        firskpoint = -1

    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Neighbors loaded")
    print("     Eigenvalues loaded")
    # print(d.eigenvalues)

    connections = np.load("dp.npy")
    print("     Modulus of direct product loaded")

    print()
    print("     Finished reading data")
    print()
    ##########################################################################
    bands = np.full((d.nks, d.nbnd, 10), -2, dtype=int)
    for bnd in range(d.nbnd):
        bands[:, bnd, 0] = bnd  # Initial attribution of band numbers

    ##########################################################################
    TOL = 0.9
    ntentative = 2  # Nr of tentatives (doesn't seem useful in this configuration
    initialks = []  # List to save the initial k-points that are choosen randomly
    # signal = nr of matches, -1 if contradictory matches, band = real band
    signal = np.zeros((d.nks, d.nbnd, ntentative), dtype=int)
    # Create arrays of tentatives
    for tentative in range(1, ntentative):
        print("     Tentative:", tentative)
        if 0 <= firskpoint < d.nks and tentative == 1:  # If initial k-point is given
            kp0 = firskpoint
        else:
            kp0 = randrange(d.nks)  # Chooses a k-point randomly
        print("     starting k-point:", kp0)
        initialks.append(kp0)  # Stores the departure k-point for future reference
        listdone = []  # List to store the k-points that have been analysed
        listk = [kp0]
        # initializes 1st k-point of the series
        bands[kp0, :, tentative] = bands[kp0, :, 0]
        signal[kp0, :, 1] = 1  # First k-point has signal 1

        while len(listk) > 0:  # Runs through the list of neighbors not already done
            nk = listk[0]  # Chooses the first from the list
            for i in range(4):  # Calculates the four points around
                for b1, b2 in itertools.product(range(d.nbnd), range(d.nbnd)):
                    # Finds connections between k-points/bands
                    if connections[nk, i, b1, b2] > TOL:
                        for bb in range(d.nbnd):
                            if bands[nk, bb, tentative] == b1:
                                if bands[d.neighbors[nk, i], bb, tentative] == -2:
                                    bands[d.neighbors[nk, i], bb, tentative] = b2
                                    signal[
                                        d.neighbors[nk, i], bb, tentative
                                    ] += 1  # Signal a connection
                                elif bands[d.neighbors[nk, i], bb, tentative] == b2:
                                    signal[d.neighbors[nk, i], bb, tentative] += 1
                                else:  # Signal a contradiction
                                    signal[d.neighbors[nk, i], bb, tentative] = -1
                                    print(d.neighbors[nk, i], b2, "  signal = -1")
                                break

                if (
                    d.neighbors[nk, i] not in listk
                    and d.neighbors[nk, i] not in listdone
                    and d.neighbors[nk, i] != -1
                ):
                    listk.append(d.neighbors[nk, i])
            ##           print(nk,i,bands[d.neighbors[nk,i],:,1])
            listk.remove(nk)  # Remove k-point from the list of todo
            listdone.append(nk)  # Add k-point to the list of done

    #   SEP = " "
    #   print("     Bands: gives the machine nr that belongs to new band (nk,nb)")
    #   for nb in range(6):
    #       nk = -1
    #       print()
    #       print("  New band " + str(nb))
    #       bands_numbers(d.nkx, d.nky, bands[:, nb, 1])
    #   print()
    #   for nb in range(6):
    #       nk = -1
    #       print()
    #       print("  Signal " + str(nb))
    #       bands_numbers(d.nkx, d.nky, signal[:, nb, 1])
    #   print()
    #   with open("bandsfinal.npy", "wb") as f:
    #       np.save(f, bands[:,:,1])
    #   f.close()
    #   with open("signalfinal.npy", "wb") as f:
    #       np.save(f, signal[:,:,1])
    #   f.close()
    #   sys.exit("Stop")
    ##########################################################################
    print()
    print("     Correcting problems found using more relaxed condition.")
    print()
    ##########################################################################

    kp0 = randrange(d.nks)  # Chooses a k-point randomly
    print("     Starting k-point for the relaxed condition:", kp0)
    initialks.append(kp0)  # Stores the departure k-point for future reference
    listdone = []  # List to store the k-points that have been analysed
    listk = [kp0]
    attcount = 0

    while len(listk) > 0:  # Runs through the list of neighbors not already done
        nk = listk[0]  # Chooses the first from the list
        for i in range(4):  # Calculates the four points around
            for b1, b2 in itertools.product(range(d.nbnd), range(d.nbnd)):
                # Finds connections between k-points/bands
                if connections[nk, i, b1, b2] > 0.8:
                    for bb in range(d.nbnd):
                        if bands[nk, bb, tentative] == b1:
                            if bands[d.neighbors[nk, i], bb, tentative] == -2:
                                bands[d.neighbors[nk, i], bb, tentative] = b2
                                signal[
                                    d.neighbors[nk, i], bb, tentative
                                ] += 1  # Signal a connection
                                attcount += 1
                            break

            if (
                d.neighbors[nk, i] not in listk
                and d.neighbors[nk, i] not in listdone
                and d.neighbors[nk, i] != -1
            ):
                listk.append(d.neighbors[nk, i])
        ##           print(nk,i,bands[d.neighbors[nk,i],:,1])
        listk.remove(nk)  # Remove k-point from the list of todo
        listdone.append(nk)  # Add k-point to the list of done

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
                d.neighbors[kpb1b2[i, 0], :],
            )
            print(bands[kpb1b2[i, 0], :, 1])
            validneig = np.count_nonzero(d.neighbors[kpb1b2[i, 0], :] != -1)
            count11 = count12 = count21 = count22 = 0
            for neig in range(4):
                if d.neighbors[kpb1b2[i, 0], neig] != -1:
                    print(bands[d.neighbors[kpb1b2[i, 0], neig], :, 1])
                    print(
                        kpb1b2[i, 0],
                        d.neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2],
                        bnproblem[i * 2],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        d.neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2],
                        bnproblem[i * 2 + 1],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2], bnproblem[i * 2 + 1]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        d.neighbors[kpb1b2[i, 0], neig],
                        bnproblem[i * 2 + 1],
                        bnproblem[i * 2],
                        connections[
                            kpb1b2[i, 0], neig, bnproblem[i * 2 + 1], bnproblem[i * 2]
                        ],
                    )
                    print(
                        kpb1b2[i, 0],
                        d.neighbors[kpb1b2[i, 0], neig],
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
    for c, nk0, nb0 in itertools.product(range(3), range(d.nks), range(d.nbnd)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = d.neighbors[nk0, neig]
                for nb1 in range(d.nbnd):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.01,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(d.nks), range(d.nbnd)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = d.neighbors[nk0, neig]
                for nb1 in range(d.nbnd):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.01,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.02,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(d.nks), range(d.nbnd)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = d.neighbors[nk0, neig]
                for nb1 in range(d.nbnd):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.01,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.02,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.03,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    for nk0, nb0 in itertools.product(range(d.nks), range(d.nbnd)):
        if bands[nk0, nb0, 1] == -1:
            for neig in range(4):
                nk1 = d.neighbors[nk0, neig]
                for nb1 in range(d.nbnd):
                    if (
                        nk1 != -1
                        and bands[nk1, nb1, 1] != -1
                        and connections[nk0, neig, nb0, nb1] > 0.6
                    ):
                        if math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.01,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.02,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.04,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break
                        elif math.isclose(
                            d.eigenvalues[nk0, nb0],
                            d.eigenvalues[nk1, nb1],
                            abs_tol=0.08,
                        ):
                            if np.all(bands[nk0, :, 1] != bands[nk1, nb1, 1]):
                                bands[nk0, nb0, 1] = bands[nk1, nb1, 1]
                                if signal[nk0, nb0, 1] == 0:
                                    signal[nk0, nb0, 1] = 1
                                eigcont += 1
                                break

    ##########################################################################
    bandsfinal = np.full((d.nks, d.nbnd), -1, dtype=int)  # Array for the final results
    # gives the machine band that belongs to band (nk,nb)
    signalfinal = np.zeros((d.nks, d.nbnd), dtype=int)  # Array for final signalling
    first = True
    attrib = []
    merger = 1
    cont = 1
    for i in initialks:  # Runs through all sets of bands
        if first:  # First set is special
            first = False
            bandsfinal = bands[:, :, 1]  # Starts with the first set
            signalfinal = signal[:, :, 1].astype(int)  # Starts with first set signaling
            for nk in range(d.nks):
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
                    for nk in range(d.nks):
                        for nb in range(d.nbnd):
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
    nrnotattrib = np.full((d.nbnd), -1, dtype=int)
    SEP = " "
    print("     Bands: gives the machine nr that belongs to new band (nk,nb)")
    for nb in range(d.nbnd):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(bandsfinal[:, nb] == -2)
        print()
        print("  New band " + str(nb) + "       nr of fails: " + str(nrnotattrib[nb]))
        bands_numbers(d.nkx, d.nky, bandsfinal[:, nb])
    print()
    print(" Signaling: how many events in each band signaled.")
    nrsignal = np.full((d.nbnd, 8), -2, dtype=int)
    for nb in range(d.nbnd):
        nk = -1
        nrsignal[nb, 0] = str(np.count_nonzero(signalfinal[:, nb] == -2))
        nrsignal[nb, 1] = str(np.count_nonzero(signalfinal[:, nb] == -1))
        nrsignal[nb, 2] = str(np.count_nonzero(signalfinal[:, nb] == 0))
        nrsignal[nb, 3] = str(np.count_nonzero(signalfinal[:, nb] == 1))
        nrsignal[nb, 4] = str(np.count_nonzero(signalfinal[:, nb] == 2))
        nrsignal[nb, 5] = str(np.count_nonzero(signalfinal[:, nb] == 3))
        nrsignal[nb, 6] = str(np.count_nonzero(signalfinal[:, nb] == 4))
        nrsignal[nb, 7] = str(np.count_nonzero(signalfinal[:, nb] == 5))
        print()
        print(
            "     "
            + str(nb)
            + "         -2: "
            + str(nrsignal[nb, 0])
            + "         -1: "
            + str(nrsignal[nb, 1])
            + "     0: "
            + str(nrsignal[nb, 2])
        )
        bands_numbers(d.nkx, d.nky, signalfinal[:, nb])

    print()
    print("     Resume of results")
    print()
    print("     nr k-points not attributed to a band (bandfinal=-2)")
    print("     Band       nr k-points")
    for nb in range(d.nbnd):
        print("     ", nb, "         ", nrnotattrib[nb])

    print()
    print("     Signaling: how many events in each band signaled.")
    print("     Band    -2   -1    0    1    2    3    4    5")
    for nb in range(d.nbnd):
        print("      " + str(nb) + "   " + str(nrsignal[nb, :]))

    print()
    print("     Continuity recovered: ", negcount)
    print("     Found by eigenvalue continuity: ", eigcont)
    print("     More relaxed attribution: ", attcount)
    print("     Merged ", merger, " sets")
    print()

    print("     Bands not usable (not completed) as they are")
    for nb in range(d.nbnd):
        if nrsignal[nb, 0] != 0: 
            print(
                "      band ", nb, "  failed attribution of ", nrsignal[nb, 0], " k-points"
            )
        if nrsignal[nb, 1] != 0: 
            print(
                "      band ", nb, "  has incongruences in ", nrsignal[nb, 1], " k-points"
            )
        if nrsignal[nb, 2] != 0:
            print(
                "      band ", nb, "  signals 0 in", nrsignal[nb, 2], " k-points"
            )
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
