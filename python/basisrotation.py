"""
  This program finds the problematic cases and makes a basis rotation of
  the wavefunctions
"""

from log_libs import log
import loaddata as d

LOG = log('basisrotation', 'BASIS ROTATION', d.version)

import sys
import time

import numpy as np
from scipy.optimize import minimize

# These are the subroutines and functions
from headerfooter import header, footer
from write_k_points import _bands_numbers
from clustering_libs import evaluate_result
from cli import basisrotation_cli

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
        LOG.info(f'old_dp: {connections[k, i_neig, machbn, bneig]} new_dp: {dp}')
        dot_products.append(dp)

        dot_products_neigs = []
        for j_neig, k2_neig in enumerate(d.neighbors[kneig]):
            if k2_neig == -1:
                continue
            bn2neig = bnfinal[k2_neig, bn]
            connection = dp if k2_neig == k else connections[kneig, j_neig, bneig, bn2neig]
            dot_products_neigs.append(connection)
        new_signal = evaluate_result(dot_products_neigs)
        LOG.info(f'old_signal: {sigfinal[kneig, bn]} new_signal: {new_signal}')
        sigfinal[kneig, bn] = new_signal
    
    sigfinal[k, bn] = evaluate_result(dot_products)

    return signalfinal

###################################################################################

if __name__ == "__main__":
    LOG.header()

    STARTTIME = time.time()  # Starts counting time

    args = basisrotation_cli()
    lastband = args['MAX_BAND']

    # Reading data needed for the run
    berrypath = d.berrypath
    LOG.info("     Unique reference of run:", d.refname)
    LOG.info("     Path to BERRY files:", berrypath)
    LOG.info("     Directory where the wfc are:", d.wfcdirectory)
    LOG.info("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    LOG.info("     Total number of k-points:", d.nks)
    LOG.info("     Total number of points in real space:", d.nr)
    LOG.info("     Number of bands:", d.nbnd)
    LOG.info()
    LOG.info("     Neighbors loaded")
    LOG.info("     Eigenvalues loaded")

    dotproduct = np.load("dpc.npy")
    connections = np.load("dp.npy")
    LOG.info("     Dot product loaded")

    LOG.info("     Reading files bandsfinal.npy and signalfinal.npy")
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    with open("degeneratefinal.npy", "rb") as fich:
        degeneratefinal = np.load(fich)
    fich.close()

    ###################################################################################
    LOG.info()
    LOG.info("     **********************")
    LOG.info("     Problems not solved")
    if degeneratefinal.shape[0] == 0:
        LOG.footer()
        exit()
    # Consider just the ones below last band wanted
    kpproblem = degeneratefinal[:, 0]
    bnproblem = degeneratefinal[:, [1, 2]]
    bands_use = np.logical_and(bnproblem[:, 0] <= lastband, bnproblem[:, 1] <= lastband)
    kpproblem = kpproblem[bands_use]
    bnproblem = bnproblem[bands_use]
    machbandproblem = np.array(list(zip(bandsfinal[kpproblem, bnproblem[:, 0]],
                                        bandsfinal[kpproblem, bnproblem[:, 1]])))
    LOG.info("      k-points", kpproblem)
    LOG.info("      in bands", bnproblem)
    LOG.info("      mach  bands", machbandproblem)

#   LOG.info(karray)
#   LOG.info(karray[0])
#   LOG.info(len(karray))

    for nki, nk0 in enumerate(kpproblem):
        LOG.info()
        LOG.info()
        LOG.info("     K-point where problem will be solved:", nk0)
        for j in range(4):  # Find the neigbhors of the k-point to be used
            nk = d.neighbors[nk0, j]  # on interpolation
#            LOG.info(j, nk, bnproblem[karray[i][0]])
            if nk != -1 and signalfinal[nk,bnproblem[nki, 0]] > DEGENERATE and signalfinal[nk,bnproblem[nki, 1]] > DEGENERATE:
                nb1 = machbandproblem[nki, 0]
                nb2 = machbandproblem[nki, 1]
                nkj = j
                break

        LOG.info("     Reference k-point:", nk)
        LOG.info("     Bands that will be mixed:",nb1, nb2)

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

        LOG.info()

        myoptions={'disp':False}
        bnds = ((-1, 1), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1), (-np.pi, np.pi), (-np.pi, np.pi))

        res = minimize(func, a, args=dot, options = myoptions, bounds=bnds, constraints=const)

        LOG.info("     Result:", res.x)
        ca1 = res.x[0]*np.exp(1j*res.x[1])
        LOG.info("     a1 = ", ca1)
        ca2 = np.sqrt(1 - res.x[0]**2)*np.exp(1j*res.x[2])
        LOG.info("     a2 = ", ca2)
        cb1 = res.x[3]*np.exp(1j*res.x[4])
        LOG.info("     b1 = ", cb1)
        cb2 = np.sqrt(1 - res.x[3]**2)*np.exp(1j*res.x[5])
        LOG.info("     b2 = ", cb2)

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
        LOG.info()
        LOG.info("     Reading file: ", infile)
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
        LOG.info("     Reading file: ", infile)
        with open(infile, "rb") as f:  # Load the wfc from file
            psi2[:] = np.load(f)  # puts wfc in this array
        f.close()

        psinewA = psi1*ca1 + psi2*ca2
        psinewB = psi1*cb1 + psi2*cb2

        signalfinal = set_new_signal(nk0, nb1, psinewA, bandsfinal, signalfinal, connections)
        signalfinal = set_new_signal(nk0, nb2, psinewB, bandsfinal, signalfinal, connections)

        # Save new files
        LOG.info()
        outfile = (
            d.wfcdirectory
            + "k0"
            + str(nk0)
            + "b0"
            + str(nb1)
            + ".wfc1"
        )
        LOG.info("     Writing file: ", outfile)
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
        LOG.info("     Writing file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewB)
        f.close()


    #sys.exit("Stop")
    ###################################################################################
    LOG.info()
    LOG.info(" *** Final Report ***")
    LOG.info()
    nrnotattrib = np.full((d.nbnd), -1, dtype=int)
    SEP = " "
    #LOG.info("     Bands: gives the original band that belongs to new band (nk,nb)")
    for nb in range(lastband + 1):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(signalfinal[:, nb] == NOT_SOLVED)
        LOG.info()
        LOG.info(
            "  New band "
            + str(nb)
            + "         nr of fails: "
            + str(nrnotattrib[nb])
        )
        _bands_numbers(d.nkx, d.nky, bandsfinal[:, nb])
    LOG.info()
    LOG.info(" Signaling")
    nrsignal = np.zeros((d.nbnd, CORRECT+1), dtype=int)
    for nb in range(lastband + 1):
        nk = -1
        for s in range(CORRECT+1):
            nrsignal[nb, s] = np.count_nonzero(signalfinal[:, nb] == s)

        LOG.info()
        LOG.info(
            "     "
            + str(nb)
            + f"         {NOT_SOLVED}: "
            + str(nrsignal[nb, NOT_SOLVED])
        )
        _bands_numbers(d.nkx, d.nky, signalfinal[:, nb])

    LOG.info()
    LOG.info("     Resume of results")
    LOG.info()
    LOG.info(f"     nr k-points not attributed to a band (bandfinal={NOT_SOLVED})")
    LOG.info("     Band       nr k-points")
    for nb in range(lastband + 1):
        LOG.info("     ", nb, "         ", nrnotattrib[nb])

    LOG.info()
    LOG.info("     Signaling")

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
    LOG.info(signal_report)

    LOG.info()

    LOG.info("     Bands not usable (not completed)")
    for nb in range(lastband + 1):
        if nrsignal[nb, NOT_SOLVED] != 0:
            LOG.info(
                "      band ", nb, "  failed attribution of ", nrsignal[nb, NOT_SOLVED], " k-points"
            )
        if nrsignal[nb, MISTAKE] != 0:
            LOG.info(
                "      band ", nb, "  has incongruences in ", nrsignal[nb, MISTAKE], " k-points"
            )
        if nrsignal[nb, POTENTIAL_MISTAKE] != 0:
            LOG.info(
                "      band ", nb, f"  signals {POTENTIAL_MISTAKE} in", nrsignal[nb, POTENTIAL_MISTAKE], " k-points"
            )
    LOG.info()

    with open('signalfinal.npy', 'wb') as f:
        np.save(f, signalfinal)

    ###################################################################################
    # Finished
    LOG.footer()
