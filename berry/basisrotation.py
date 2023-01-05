"""
  This program finds the problematic cases and makes a basis rotation of
  the wavefunctions
"""
import os
import logging

from scipy.optimize import minimize

import numpy as np

from berry import log
from berry._subroutines.write_k_points import _bands_numbers
from berry._subroutines.clustering_libs import evaluate_result

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


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

def set_new_signal(k, bn, psinew, bnfinal, sigfinal, connections, logger: log):
    machbn = bnfinal[k, bn]

    dot_products = []
    for i_neig, kneig in enumerate(d.neighbors[k]):
        if kneig == -1:
            continue

        bneig = bnfinal[kneig, bn]
        psineig = np.load(os.path.join(m.wfcdirectory, f"k0{kneig}b0{bneig}.wfc"))

        dphase = d_phase[:, k] * np.conjugate(d_phase[:, kneig])
        dot_product = np.sum(dphase * psinew * np.conjugate(psineig)) / m.nr
        dp = np.abs(dot_product)
        logger.info(f'\told_dp: {connections[k, i_neig, machbn, bneig]} new_dp: {dp}')
        dot_products.append(dp)

        dot_products_neigs = []
        for j_neig, k2_neig in enumerate(d.neighbors[kneig]):
            if k2_neig == -1:
                continue
            bn2neig = bnfinal[k2_neig, bn]
            connection = dp if k2_neig == k else connections[kneig, j_neig, bneig, bn2neig]
            dot_products_neigs.append(connection)
        new_signal = evaluate_result(dot_products_neigs)
        logger.info(f'\told_signal: {sigfinal[kneig, bn]} new_signal: {new_signal}')
        sigfinal[kneig, bn] = new_signal
    
    sigfinal[k, bn] = evaluate_result(dot_products)

    return signalfinal


def run_basis_rotation(max_band: int, npr: int = 1, logger_name: str = "basis", logger_level: int = logging.INFO, flush: bool = False):
    global signalfinal, d_phase
    logger = log(logger_name, "BASIS ROTATION", level=logger_level, flush=flush)

    logger.header()

    # Reading data needed for the run
    berrypath = m.berrypath
    logger.info("\tUnique reference of run:", m.refname)
    logger.info("\tPath to BERRY files:", berrypath)
    logger.info("\tDirectory where the wfc are:", m.wfcdirectory)
    logger.info("\tNumber of k-points in each direction:", m.nkx, m.nky, m.nkz)
    logger.info("\tTotal number of k-points:", m.nks)
    logger.info("\tTotal number of points in real space:", m.nr)
    logger.info("\tNumber of bands:", m.nbnd)
    logger.info()
    logger.info("\tNeighbors loaded")
    logger.info("\tEigenvalues loaded")

    d_phase = np.load(os.path.join(m.workdir, "phase.npy"))
    logger.info("\tPhases loaded")

    dotproduct = np.load(os.path.join(m.workdir, "dpc.npy"))
    connections = np.load(os.path.join(m.workdir, "dp.npy"))
    logger.info("\tDot product loaded")

    logger.info("\tReading files bandsfinal.npy and signalfinal.npy")
    bandsfinal = np.load(os.path.join(m.workdir, "bandsfinal.npy"))
    signalfinal = np.load(os.path.join(m.workdir, "signalfinal.npy"))
    degeneratefinal = np.load(os.path.join(m.workdir, "degeneratefinal.npy"))

    ###################################################################################
    logger.info()
    logger.info("\t**********************")
    logger.info("\n\tProblems not solved")
    if degeneratefinal.shape[0] == 0:
        logger.footer()
        exit(0)
    # Consider just the ones below last band wanted
    kpproblem = degeneratefinal[:, 0]
    bnproblem = degeneratefinal[:, [1, 2]]
    S1 = signalfinal[kpproblem, bnproblem[:, 0]]
    S2 = signalfinal[kpproblem, bnproblem[:, 1]]
    bands_use1 = np.logical_and(bnproblem[:, 0] <= max_band, bnproblem[:, 1] <= max_band)
    bands_use2 = np.logical_and(S1 == DEGENERATE, S2 == DEGENERATE)
    bands_use = np.logical_and(bands_use1, bands_use2)
    if np.sum(bands_use) == 0:
        logger.footer()
        exit(0)
    kpproblem = kpproblem[bands_use]
    bnproblem = bnproblem[bands_use]
    machbandproblem = np.array(list(zip(bandsfinal[kpproblem, bnproblem[:, 0]],
                                        bandsfinal[kpproblem, bnproblem[:, 1]])))

    list_str = lambda l:' (' + ', '.join(map(str, l)) + ') '
    logger.info("\tk-points\n\t", ', '.join(map(str, kpproblem)))
    logger.info("\tin bands\n\t", ', '.join(map(list_str, bnproblem)))
    logger.info("\tmatch  bands\n\t", ', '.join(map(list_str, machbandproblem)))

#   logger.info(karray)
#   logger.info(karray[0])
#   logger.info(len(karray))

    for nki, nk0 in enumerate(kpproblem):
        logger.info()
        logger.info()
        logger.info("\tK-point where problem will be solved:", nk0)
        for j in range(4):  # Find the neigbhors of the k-point to be used
            nk = d.neighbors[nk0, j]  # on interpolation
#            logger.info(j, nk, bnproblem[karray[i][0]])
            if nk != -1 and signalfinal[nk,bnproblem[nki, 0]] > DEGENERATE and signalfinal[nk,bnproblem[nki, 1]] > DEGENERATE:
                nb1 = machbandproblem[nki, 0]
                nb2 = machbandproblem[nki, 1]
                nkj = j
                break

        logger.info("\tReference k-point:", nk)
        logger.info("\tBands that will be mixed:",nb1, nb2)

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

        logger.info()

        myoptions={'disp':False}
        bnds = ((-1, 1), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1), (-np.pi, np.pi), (-np.pi, np.pi))

        res = minimize(func, a, args=dot, options = myoptions, bounds=bnds, constraints=const)

        logger.info("\tResult:", res.x)
        ca1 = res.x[0]*np.exp(1j*res.x[1])
        logger.info("\ta1 = ", ca1)
        ca2 = np.sqrt(1 - res.x[0]**2)*np.exp(1j*res.x[2])
        logger.info("\ta2 = ", ca2)
        cb1 = res.x[3]*np.exp(1j*res.x[4])
        logger.info("\tb1 = ", cb1)
        cb2 = np.sqrt(1 - res.x[3]**2)*np.exp(1j*res.x[5])
        logger.info("\tb2 = ", cb2)

        psinewA = np.zeros((int(d.nr)), dtype=complex)
        psinewB = np.zeros((int(d.nr)), dtype=complex)

        infile = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb1}.wfc")
        logger.info()
        logger.info("\tReading file: ", infile)
        psi1 = np.load(infile)  # puts wfc in this array
        infile = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb2}.wfc")
        logger.info("\tReading file: ", infile)
        psi2 = np.load(infile)  # puts wfc in this array

        psinewA = psi1*ca1 + psi2*ca2
        psinewB = psi1*cb1 + psi2*cb2

        signalfinal = set_new_signal(nk0, nb1, psinewA, bandsfinal, signalfinal, connections, logger)
        signalfinal = set_new_signal(nk0, nb2, psinewB, bandsfinal, signalfinal, connections, logger)

        # Save new files
        logger.info()
        outfile = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb1}.wfc1")
        logger.info("\tWriting file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewA)
        outfile = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb2}.wfc1")
        logger.info("\tWriting file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewB)


    #sys.exit("Stop")
    ###################################################################################
    logger.info()
    logger.info("\t*** Final Report ***")
    logger.info()
    nrnotattrib = np.full((d.nbnd), -1, dtype=int)
    SEP = " "
    #logger.info("Bands: gives the original band that belongs to new band (nk,nb)")
    for nb in range(max_band + 1):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(signalfinal[:, nb] == NOT_SOLVED)
        logger.debug()
        logger.debug(f"\tNew band {nb}\t\tnr of fails: {nrnotattrib[nb]}")
        logger.debug(_bands_numbers(d.nkx, d.nky, bandsfinal[:, nb]))
    logger.debug()
    logger.debug("\tSignaling")
    nrsignal = np.zeros((d.nbnd, CORRECT+1), dtype=int)
    for nb in range(max_band + 1):
        nk = -1
        for s in range(CORRECT+1):
            nrsignal[nb, s] = np.count_nonzero(signalfinal[:, nb] == s)

        logger.debug()
        logger.debug(f"\t{nb}\t\t{NOT_SOLVED}: {nrsignal[nb, NOT_SOLVED]}")
        logger.debug(_bands_numbers(d.nkx, d.nky, signalfinal[:, nb]))

    logger.info()
    logger.info("\tResume of results")
    logger.info()
    logger.info(f"\tnr k-points not attributed to a band (bandfinal={NOT_SOLVED})")
    logger.info("\tBand\tnr k-points")
    for nb in range(max_band + 1):
        logger.info("\t", nb, "\t", nrnotattrib[nb])

    logger.info()
    logger.info("\tSignaling")

    signal_report = '\tBands | '
    for signal in range(CORRECT+1):
        n_spaces = len(str(np.max(nrsignal[:, signal])))-1
        signal_report += ' '*n_spaces+str(signal) + '   '
    
    signal_report += '\n\t'+'-'*len(signal_report)

    for nb in range(max_band + 1):
        signal_report += f'\n\t{nb}{" "*(4-len(str(nb)) + 1)} |' + ' '
        for signal, value in enumerate(nrsignal[nb]):
                n_max = len(str(np.max(nrsignal[:, signal])))
                n_spaces = n_max - len(str(value))
                signal_report += ' '*n_spaces+str(value) + '   '
    logger.info(signal_report)

    logger.info()

    logger.info("\tBands not usable (not completed)")
    for nb in range(max_band + 1):
        if nrsignal[nb, NOT_SOLVED] != 0:
            logger.info(f"\tband {nb} failed attribution of {nrsignal[nb, NOT_SOLVED]} k-points")
        if nrsignal[nb, MISTAKE] != 0:
            logger.info(f"\tband {nb} has incongruences in {nrsignal[nb, MISTAKE]} k-points")
        if nrsignal[nb, POTENTIAL_MISTAKE] != 0:
            logger.info(f"\tband {nb} signals {POTENTIAL_MISTAKE} in {nrsignal[nb, POTENTIAL_MISTAKE]} k-points")
    logger.info()

    np.save(os.path.join(d.workdir, 'signalfinal.npy'), signalfinal)

    ###################################################################################
    # Finished
    ###################################################################################
    logger.footer()


if __name__ == "__main__":
    run_basis_rotation(9, log("basisrotation", "BASIS ROTATION", "version", logging.DEBUG))