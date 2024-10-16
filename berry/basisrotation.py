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
    # < Psi_A | Psi_1 > a_1 + < Psi_A | Psi_2 > a_2
    r1 = complex(ddot[0],ddot[1])*aa[0]*np.exp(1j*aa[1]) + complex(ddot[2],ddot[3])*np.sqrt(1 - aa[0]**2)*np.exp(1j*aa[2])
    # < Psi_B | Psi_1 > b_1 + < Psi_B | Psi_2 > b_2
    r2 = complex(ddot[4],ddot[5])*aa[3]*np.exp(1j*aa[4]) + complex(ddot[6],ddot[7])*np.sqrt(1 - aa[3]**2)*np.exp(1j*aa[5])
    # return the negative, so it minimizes instead of maximizes
    return -np.absolute(r1) - np.absolute(r2)

def set_new_signal(k, bn, psinew, bnfinal, sigfinal, connections, logger: log):
    """Verifies the result of the basis rotation, and make the necessary changes to the signal's file.
    k              the k-point where transformation was performed
    bn             array with the original band numbers for each k-point, state
    psinew         new wavefunction
    bnfinal        the band where the wavefunction ends up
    sigfinal       the new signaling of the state
    connections    the array with the original modulus of the dot products
    """
    logger.info("")
    machbn = bnfinal[k, bn]      # Original band attribution for k-point k and state bn

    dot_products = []
    for i_neig, kneig in enumerate(d.neighbors[k]):  # Run over neighbors
        if kneig == -1:    # Exclude k-points out of the set
            continue

        bneig = bnfinal[kneig, bn]   # Original band attribution of the neighbor
        logger.info(f"\tCalculating dot product <{k},{machbn + initial_band}|{kneig},{bneig + initial_band}>")
        # Get neighbor's wavefunction
        psineig = np.load(os.path.join(m.wfcdirectory, f"k0{kneig}b0{bneig + initial_band}.wfc"))

        # Calculate the dot product between the neighbor and the new state
        dphase = d_phase[:, k] * np.conjugate(d_phase[:, kneig])
        dot_product = np.sum(dphase * psinew * np.conjugate(psineig)) / m.nr
        dp = np.abs(dot_product)
        logger.info(f'\told dp: {connections[k, i_neig, machbn, bneig]} new dp: {dp}')
        dot_products.append(dp)

        # Evaluates the new signaling of the state
        dot_products_neigs = []
        for j_neig, k2_neig in enumerate(d.neighbors[kneig]):
            if k2_neig == -1:
                continue
            bn2neig = bnfinal[k2_neig, bn]
            connection = dp if k2_neig == k else connections[kneig, j_neig, bneig, bn2neig]
            dot_products_neigs.append(connection)
        new_signal = evaluate_result(dot_products_neigs)
        logger.info(f'\told signal: {sigfinal[kneig, bn]} new signal: {new_signal}')
        sigfinal[kneig, bn] = new_signal
    
    sigfinal[k, bn] = evaluate_result(dot_products)

    return signalfinal

# Start run of basis rotation
def run_basis_rotation(max_band: int, npr: int = 1, logger_name: str = "basis", logger_level: int = logging.INFO, flush: bool = False):
    global signalfinal, d_phase, initial_band
    logger = log(logger_name, "BASIS ROTATION", level=logger_level, flush=flush)

    initial_band = m.initial_band if m.initial_band != "dummy" else 0
    max_band -= initial_band

    logger.header()

    # Reading data needed for the run
    
    logger.info("\tUnique reference of run:", m.refname)
    logger.info("\tDirectory where the wfc are:", m.wfcdirectory)
    logger.info("\tNumber of k-points in each direction:", m.nkx, m.nky, m.nkz)
    logger.info("\tTotal number of k-points:", m.nks)
    logger.info("\tTotal number of points in real space:", m.nr)
    logger.info("\tNumber of bands:", max_band)
    logger.info(f"\tBands: [{initial_band}, {max_band + initial_band}]")
    logger.info()

    if m.noncolin:
        logger.info("\n\tThis is a noncolinear calculation: basis rotation is not implemented.")
        logger.info("\tExiting.")
        logger.footer()
        exit(0)

    d_phase = np.load(os.path.join(m.data_dir, "phase.npy"))
    logger.info("\tPhases loaded")

    dotproduct = np.load(os.path.join(m.data_dir, "dpc.npy"))
    connections = np.load(os.path.join(m.data_dir, "dp.npy"))
    logger.info("\tDot product loaded")

    logger.info("\tReading files bandsfinal.npy, signalfinal.npy and degeneratefinal.npy")
    bandsfinal = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    signalfinal = np.load(os.path.join(m.data_dir, "signalfinal.npy"))
    degeneratefinal = np.load(os.path.join(m.data_dir, "degeneratefinal.npy"))

    # Finished reading data from files
    ###################################################################################
    # Start identifying states to apply rotation
    logger.info()
    logger.info("\t**********************")
    logger.info("\n\tProblems signaled:")
    if degeneratefinal.shape[0] == 0:
        logger.info("\n\tNo problems found.")
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
        logger.info("\n\tNo problems found in the band range.")
        logger.footer()
        exit(0)
    kpproblem = kpproblem[bands_use]   # Arrays already filtered
    bnproblem = bnproblem[bands_use]
    matchbandproblem = np.array(list(zip(bandsfinal[kpproblem, bnproblem[:, 0]],
                                        bandsfinal[kpproblem, bnproblem[:, 1]])))

    # list_str = lambda l:' (' + ', '.join(map(str, l)) + ') '
    list_str = lambda l:' (' + ', '.join(map(str, l + initial_band)) + ') '
    logger.info("\tk-points\n\t", ', '.join(map(str, kpproblem)))
    logger.info("\tin bands\n\t", ', '.join(map(list_str, bnproblem)))
    logger.info("\tmatch  bands\n\t", ', '.join(map(list_str, matchbandproblem)))

    # runs through the list of problems
    for nkindex, nk0 in enumerate(kpproblem):  # nkindex is the index in list kpproblem
        logger.info("\n\n\tK-point where problem will be solved:", nk0)
        for j in range(m.dimensions*2):    # Find the neigbhors of the k-point to be used on interpolation
            nk = d.neighbors[nk0, j]       # k-point number of neighbor
            if nk != -1 and signalfinal[nk,bnproblem[nkindex, 0]] > DEGENERATE and signalfinal[nk,bnproblem[nkindex, 1]] > DEGENERATE:
                nb1 = matchbandproblem[nkindex, 0]   # One of the bands of the neighbor
                nb2 = matchbandproblem[nkindex, 1]   # The other band
                nkj = j
                break                      # Found a valid neighbor, can proceed
            else:
                nb1 = -1
                nb2 = -1
                nkj = -1
        logger.info("\tReference k-point:", nk)
        if nkj == -1:
            logger.info("\tNo neighbors valid for basis rotation. Skipping.")
            continue
        else:
            logger.info("\tBands that will be mixed:", nb1 + initial_band, nb2 + initial_band)
        # k-point that has a problem: nk0
        # k-point that has clear bands A and B: nkj
        # dots products read from file
        dotA1 = dotproduct[nk0, nkj, nb1, nb1]    # < nk0,nb1 | nkj,nb1 >
        dotA2 = dotproduct[nk0, nkj, nb1, nb2]    # < nk0,nb1 | nkj,nb2 >
        dotB1 = dotproduct[nk0, nkj, nb2, nb1]    # < nk0,nb2 | nkj,nb1 >
        dotB2 = dotproduct[nk0, nkj, nb2, nb2]    # < nk0,nb2 | nkj,nb2 >
        # Create array with the dot products, real and imaginary part separated, to be the parameters of func()
        dot = np.array([np.real(dotA1), np.imag(dotA1), np.real(dotA2), np.imag(dotA2), np.real(dotB1), np.imag(dotB1), np.real(dotB2), np.imag(dotB2)])
        
        # Starting values for the variables we want to find
        a1 = 0.5                 # Modulus of a_1
        a1o = 0                  # Phase of a_1
        a2 = np.sqrt(1 - a1**2)  # Modulus of a_2, is related to modulus of a_1 due to normalization
        a2o = 0                  # Phase of a_2

        b1 = 0.5                 # Modulus of b_1
        b1o = 0                  # Phase of b_1
        b2 = np.sqrt(1 - b1**2)  # Modulus of b_2, is related to modulus of b_1 due to normalization
        b2o = 0                  # Phase of b_2
        # Array with the initial values
        a = np.array([a1, a1o, a2o, b1, b1o, b2o])

        # The following is equivalent to the constraint a_1^*b_1 + a_2^*b_2 = 0
        const = ({'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.cos(a[4] - a[1]) + np.sqrt(1 - a[0]**2)*np.sqrt(1 - a[3]**2)*np.cos(a[5] - a[2])},
                 {'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.sin(a[4] - a[1]) + np.sqrt(1 - a[0]**2)*np.sqrt(1 - a[3]**2)*np.sin(a[5] - a[2])})

        logger.info()

        myoptions = {'disp': False}
        # Bounds for the variables we want to find (modulus vary between 0 and 1 and phases between -pi and +pi)
        bnds = ((0, 1), (-np.pi, np.pi), (-np.pi, np.pi), (0, 1), (-np.pi, np.pi), (-np.pi, np.pi))
        # Finds the arguments a that minimizes func() with the constraint const = 0
        res = minimize(func, a, args=dot, options = myoptions, bounds = bnds, constraints = const)

        logger.info("\tResult output:", res.x)
        ca1 = res.x[0]*np.exp(1j*res.x[1])
        logger.info("\ta1 = ", ca1)
        ca2 = np.sqrt(1 - res.x[0]**2)*np.exp(1j*res.x[2])
        logger.info("\ta2 = ", ca2)
        cb1 = res.x[3]*np.exp(1j*res.x[4])
        logger.info("\tb1 = ", cb1)
        cb2 = np.sqrt(1 - res.x[3]**2)*np.exp(1j*res.x[5])
        logger.info("\tb2 = ", cb2)
        verification = np.conjugate(ca1)*cb1 + np.conjugate(ca2)*cb2
        logger.info(f"\tVerification, should be zero: {verification}")

        # Create arrays for the new wavefunctions
        psinewA = np.zeros((int(m.nr)), dtype=complex)
        psinewB = np.zeros((int(m.nr)), dtype=complex)

        # Load old wavefunctions
        infile = os.path.join(m.wfcdirectory, f"k0{nk0}b0{nb1 + initial_band}.wfc")
        logger.info()
        logger.info("\tReading old wavefunction 1: ", infile)
        psi1 = np.load(infile)  # puts wfc in this array
        infile = os.path.join(m.wfcdirectory, f"k0{nk0}b0{nb2 + initial_band}.wfc")
        logger.info("\tReading old wavefunction 2: ", infile)
        psi2 = np.load(infile)  # puts wfc in this array

        # Calculate new wavefunctions
        psinewA = psi1*ca1 + psi2*ca2
        psinewB = psi1*cb1 + psi2*cb2

        signalfinal = set_new_signal(nk0, nb1, psinewA, bandsfinal, signalfinal, connections, logger)
        signalfinal = set_new_signal(nk0, nb2, psinewB, bandsfinal, signalfinal, connections, logger)

        # Save new wavefunctions to files with extension wfc1
        logger.info()
        outfile = os.path.join(m.wfcdirectory, f"k0{nk0}b0{nb1 + initial_band}.wfc1")
        logger.info("\tWriting file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewA)
        outfile = os.path.join(m.wfcdirectory, f"k0{nk0}b0{nb2 + initial_band}.wfc1")
        logger.info("\tWriting file: ", outfile)
        with open(outfile, "wb") as f:
            np.save(f, psinewB)


    #sys.exit("Stop")
    ###################################################################################
    logger.info()
    logger.info("\t*** Final Report ***")
    logger.info()
    nrnotattrib = np.full((max_band), -1, dtype=int)
    SEP = " "
    #logger.info("Bands: gives the original band that belongs to new band (nk,nb)")
    for nb in range(max_band + 1):
        nk = -1
        nrnotattrib[nb] = np.count_nonzero(signalfinal[:, nb] == NOT_SOLVED)
        logger.debug()
        logger.debug(f"\tNew band {nb + initial_band}\t\tnr of fails: {nrnotattrib[nb]}")
        logger.debug(_bands_numbers(m.nkx, m.nky, bandsfinal[:, nb]))
    logger.debug()
    logger.debug("\tSignaling")
    nrsignal = np.zeros((max_band, CORRECT+1), dtype=int)
    for nb in range(max_band + 1):
        nk = -1
        for s in range(CORRECT+1):
            nrsignal[nb, s] = np.count_nonzero(signalfinal[:, nb] == s)

        logger.debug()
        logger.debug(f"\t{nb + initial_band}\t\t{NOT_SOLVED}: {nrsignal[nb, NOT_SOLVED]}")
        logger.debug(_bands_numbers(m.nkx, m.nky, signalfinal[:, nb]))

    logger.info()
    logger.info("\tResume of results")
    logger.info()
    logger.info(f"\tnr k-points not attributed to a band (bandfinal={NOT_SOLVED})")
    logger.info("\tBand\tnr k-points")
    for nb in range(max_band + 1):
        logger.info("\t", nb + initial_band, "\t", nrnotattrib[nb])

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
            logger.info(f"\tband {nb + initial_band} failed attribution of {nrsignal[nb, NOT_SOLVED]} k-points")
        if nrsignal[nb, MISTAKE] != 0:
            logger.info(f"\tband {nb + initial_band} has incongruences in {nrsignal[nb, MISTAKE]} k-points")
        if nrsignal[nb, POTENTIAL_MISTAKE] != 0:
            logger.info(f"\tband {nb + initial_band} signals {POTENTIAL_MISTAKE} in {nrsignal[nb, POTENTIAL_MISTAKE]} k-points")
    logger.info()

    np.save(os.path.join(m.data_dir, 'signalfinal.npy'), signalfinal)

    ###################################################################################
    # Finished
    ###################################################################################
    logger.footer()


if __name__ == "__main__":
    run_basis_rotation(9, log("basisrotation", "BASIS ROTATION", "version", logging.DEBUG))