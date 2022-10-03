"""
  This program finds the problematic cases and makes a basis rotation of
  the wavefunctions
"""

import sys
import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

# These are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d
from write_k_points import _bands_numbers
from clustering_libs import evaluate_result

# pylint: disable=C0103
###################################################################################
CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

def func(aa, ddot):
    r1 = complex(ddot[0],ddot[1])*aa[0]*np.exp(1j*aa[1]) + complex(ddot[2],ddot[3])*np.sqrt(1 - aa[0]**2)*np.exp(1j*aa[2])
    r2 = complex(ddot[4],ddot[5])*aa[3]*np.exp(1j*aa[4]) + complex(ddot[6],ddot[7])*np.sqrt(1 - aa[3]**2)*np.exp(1j*aa[5])
    return -np.absolute(r1) - np.absolute(r2)

def set_new_signal(k, bn, psinew, bnfinal, sigfinal, connections):
    machbn = bnfinal[k, bn]

    dot_products = []
    for i_neig, kneig in enumerate(d.neighbors[k]):
        if kneig == -1:
            continue

        bneig = bnfinal[kneig, bn]
        infile = (
            d.wfcdirectory
            + "k0"
            + str(kneig)
            + "b0"
            + str(bneig)
            + ".wfc"
        )
        psineig = np.zeros((int(d.nr)), dtype=complex)
        with open(infile, "rb") as f:  # Load the wfc from file
            psineig[:] = np.load(f)

        dphase = d.phase[:, k] * np.conjugate(d.phase[:, kneig])
        dot_product = np.sum(dphase * psinew * np.conjugate(psineig)) / d.nr
        dp = np.abs(dot_product)
        print(f'old_dp: {connections[k, i_neig, machbn, bneig]} new_dp: {dp}')
        dot_products.append(dp)

        dot_products_neigs = []
        for j_neig, k2_neig in enumerate(d.neighbors[kneig]):
            if k2_neig == -1:
                continue
            bn2neig = bnfinal[k2_neig, bn]
            connection = dp if k2_neig == k else connections[kneig, j_neig, bneig, bn2neig]
            dot_products_neigs.append(connection)
        new_signal = evaluate_result(dot_products_neigs)
        print(f'old_signal: {sigfinal[kneig, bn]} new_signal: {new_signal}')
        sigfinal[kneig, bn] = new_signal
    
    sigfinal[k, bn] = evaluate_result(dot_products)

    return signalfinal

###################################################################################

if __name__ == "__main__":
    header("BASIS ROTATION", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) != 2:
        print(
            "     ERROR in number of arguments.\
                    You probably want to give the last band to be considered."
        )
        sys.exit("Stop")

    lastband = int(sys.argv[1])

    # Reading data needed for the run
    berrypath = d.berrypath
    print("     Unique reference of run:", d.refname)
    print("     Path to BERRY files:", berrypath)
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Total number of points in real space:", d.nr)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Neighbors loaded")
    print("     Eigenvalues loaded")

    dotproduct = np.load("dpc.npy")
    connections = np.load("dp.npy")
    print("     Dot product loaded")

    print("     Reading files bandsfinal.npy and signalfinal.npy")
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()

    ###################################################################################
    print()
    print("     **********************")
    print("     Problems not solved")
    kpproblem, bnproblem = np.where(signalfinal == DEGENERATE)
    machbandproblem = bandsfinal[kpproblem, bnproblem]
    if bnproblem.size > 0:
        while bnproblem[-1] > lastband:  # Consider just the ones below last band wanted
            bnproblem = bnproblem[:-1]
            kpproblem = kpproblem[:-1]
    machbandproblem = bandsfinal[kpproblem, bnproblem]
    error_bands = machbandproblem == -1
    machbandproblem[error_bands] = bnproblem[error_bands]
    print("      k-points", kpproblem)
    print("      in bands", bnproblem)
    print("      mach  bands", machbandproblem)

    karray = [np.where(kpproblem == element)[0].tolist() for element in np.unique(kpproblem)]

#   print(karray)
#   print(karray[0])
#   print(len(karray))

    for i in range(len(karray)):
        nk0 = kpproblem[karray[i][0]]
        print()
        print()
        print("     K-point where problem will be solved:", nk0)
        for j in range(4):  # Find the neigbhors of the k-point to be used
            nk = d.neighbors[nk0, j]  # on interpolation
#            print(j, nk, bnproblem[karray[i][0]])
            if nk != -1 and signalfinal[nk,bnproblem[karray[i][0]]] > DEGENERATE and signalfinal[nk,bnproblem[karray[i][1]]] > DEGENERATE:
                nb1 = machbandproblem[karray[i][0]]
                nb2 = machbandproblem[karray[i][1]]
                nkj = j
                break

        print("     Reference k-point:", nk)
        print("     Bands that will be mixed:",nb1, nb2)

        dotA1 = dotproduct[nk0, nkj, nb1, nb1]
        dotA2 = dotproduct[nk0, nkj, nb1, nb2]
        dotB1 = dotproduct[nk0, nkj, nb2, nb1]
        dotB2 = dotproduct[nk0, nkj, nb2, nb2]
        dot = np.array([np.real(dotA1), np.imag(dotA1), np.real(dotA2), np.imag(dotA2), np.real(dotB1), np.imag(dotB1), np.real(dotB2), np.imag(dotB2)])
        
        a1 = 0.5
        a1o = 0
        a2 = np.sqrt(1 - a1**2)
        a2o = 0

        b1 = 0.5
        b1o = 0
        b2 = np.sqrt(1 - b1**2)
        b2o = 0

        a = np.array([a1, a1o, a2o, b1, b1o, b2o])

        const = ({'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.cos(a[4] - a[1]) + np.sqrt(1 - a1**2)*np.sqrt(1 - b1**2)*np.cos(a[5] - a[2])},
                 {'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.sin(a[4] - a[1]) + np.sqrt(1 - a1**2)*np.sqrt(1 - b1**2)*np.sin(a[5] - a[2])})

        print()

        myoptions={'disp':False}
        bnds = ((-1, 1), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1), (-np.pi, np.pi), (-np.pi, np.pi))

        res = minimize(func, a, args=dot, options = myoptions, bounds=bnds, constraints=const)

        print("     Result:", res.x)
        ca1 = res.x[0]*np.exp(1j*res.x[1])
        print("     a1 = ", ca1)
        ca2 = np.sqrt(1 - res.x[0]**2)*np.exp(1j*res.x[2])
        print("     a2 = ", ca2)
        cb1 = res.x[3]*np.exp(1j*res.x[4])
        print("     b1 = ", cb1)
        cb2 = np.sqrt(1 - res.x[3]**2)*np.exp(1j*res.x[5])
        print("     b2 = ", cb2)

        psinewA = np.zeros((int(d.nr)), dtype=complex)
        psinewB = np.zeros((int(d.nr)), dtype=complex)
        psi1 = np.zeros((int(d.nr)), dtype=complex)
        psi2 = np.zeros((int(d.nr)), dtype=complex)

        infile = (
            d.wfcdirectory
            + "k0"
            + str(nk0)
            + "b0"
            + str(nb1)
            + ".wfc"
        )
        print()
        print("     Reading file: ", infile)
        with open(infile, "rb") as f:  # Load the wfc from file
            psi1[:] = np.load(f)  # puts wfc in this array
        f.close()
        infile = (
            d.wfcdirectory
            + "k0"
            + str(nk0)
            + "b0"
            + str(nb2)
            + ".wfc"
        )
        print("     Reading file: ", infile)
        with open(infile, "rb") as f:  # Load the wfc from file
            psi2[:] = np.load(f)  # puts wfc in this array
        f.close()

        psinewA = psi1*ca1 + psi2*ca2
        psinewB = psi1*cb1 + psi2*cb2

        signalfinal = set_new_signal(nk0, nb1, psinewA, bandsfinal, signalfinal, connections)
        signalfinal = set_new_signal(nk0, nb2, psinewB, bandsfinal, signalfinal, connections)

        # Save new files
        print()
        outfile = (
            d.wfcdirectory
            + "k0"
            + str(nk0)
            + "b0"
            + str(nb1)
            + ".wfc1"
        )
        print("     Writing file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewA)
        f.close()
        outfile = (
            d.wfcdirectory
            + "k0"
            + str(nk0)
            + "b0"
            + str(nb2)
            + ".wfc1"
        )
        print("     Writing file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewB)
        f.close()


    #sys.exit("Stop")
    ###################################################################################
    print()
    print(" *** Final Report ***")
    print()
    nrnotattrib = np.full((d.nbnd), -1, dtype=int)
    SEP = " "
    #print("     Bands: gives the original band that belongs to new band (nk,nb)")
    for nb in range(lastband + 1):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(signalfinal[:, nb] == NOT_SOLVED)
        print()
        print(
            "  New band "
            + str(nb)
            + "         nr of fails: "
            + str(nrnotattrib[nb])
        )
        _bands_numbers(d.nkx, d.nky, bandsfinal[:, nb])
    print()
    print(" Signaling")
    nrsignal = np.zeros((d.nbnd, CORRECT+1), dtype=int)
    for nb in range(lastband + 1):
        nk = -1
        for s in range(CORRECT+1):
            nrsignal[nb, s] = np.count_nonzero(signalfinal[:, nb] == s)

        print()
        print(
            "     "
            + str(nb)
            + f"         {NOT_SOLVED}: "
            + str(nrsignal[nb, NOT_SOLVED])
        )
        _bands_numbers(d.nkx, d.nky, signalfinal[:, nb])

    print()
    print("     Resume of results")
    print()
    print(f"     nr k-points not attributed to a band (bandfinal={NOT_SOLVED})")
    print("     Band       nr k-points")
    for nb in range(lastband + 1):
        print("     ", nb, "         ", nrnotattrib[nb])

    print()
    print("     Signaling")

    signal_report = '    Bands | '
    for signal in range(CORRECT+1):
        n_spaces = len(str(np.max(nrsignal[:, signal])))-1
        signal_report += ' '*n_spaces+str(signal) + '   '
    
    signal_report += '\n'+'-'*len(signal_report)

    for nb in range(lastband + 1):
        signal_report += f'\n     {nb}{" "*(4-len(str(nb)))} |' + ' '
        for signal, value in enumerate(nrsignal[nb]):
                n_max = len(str(np.max(nrsignal[:, signal])))
                n_spaces = n_max - len(str(value))
                signal_report += ' '*n_spaces+str(value) + '   '
    print(signal_report)

    print()

    print("     Bands not usable (not completed)")
    for nb in range(lastband + 1):
        if nrsignal[nb, NOT_SOLVED] != 0:
            print(
                "      band ", nb, "  failed attribution of ", nrsignal[nb, NOT_SOLVED], " k-points"
            )
        if nrsignal[nb, MISTAKE] != 0:
            print(
                "      band ", nb, "  has incongruences in ", nrsignal[nb, MISTAKE], " k-points"
            )
        if nrsignal[nb, POTENTIAL_MISTAKE] != 0:
            print(
                "      band ", nb, f"  signals {POTENTIAL_MISTAKE} in", nrsignal[nb, POTENTIAL_MISTAKE], " k-points"
            )
    print()

    with open('signalfinal.npy', 'wb') as f:
        np.save(f, signalfinal)

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
