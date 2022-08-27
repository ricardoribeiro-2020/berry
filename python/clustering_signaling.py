import numpy as np
import loaddata as d
import contatempo
from headerfooter import header, footer
import time

OLD_CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3

CORRECT = 4
OTHER = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

def evaluate_point(k, bn, bnfinal):
    machbn = bnfinal[k, bn]
    energy = d.eigenvalues[k, machbn]
    kneigs = d.neighbors[d.neighbors[k] != -1]
    energies = d.eigenvalues[kneigs, :]

    d_energies = np.abs(energies - energy)
    d_min = np.min(d_energies, axis=1)

    scores = d_min/d_energies[np.arange(len(d_energies)), bnfinal[kneigs, bn]]
    score = np.mean(scores)

    TOL = 0.9
    TOL_ERROR = 0.7

    if score > TOL:
        return CORRECT
    
    if score > TOL_ERROR:
        return OTHER
    
    return MISTAKE


if __name__ == "__main__":
    header("BASIS ROTATION", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

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
    print("     Dot product loaded")

    print("     Reading files bandsfinal.npy and signalfinal.npy")
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()

    correct_signalfinal = np.copy(signalfinal)
    correct_signalfinal[signalfinal == OLD_CORRECT] = CORRECT

    ks_pC, bnds_pC = np.where(signalfinal == POTENTIAL_CORRECT)
    ks_pM, bnds_pM = np.where(signalfinal == POTENTIAL_MISTAKE)

    ks = np.concatenate(ks_pC, ks_pM)
    bnds = np.concatenate(bnds_pC, bnds_pM)

    for k, bn in zip(ks, bnds):
        correct_signalfinal[k, bn] = evaluate_point(k, bn, bandsfinal)
    
    final_report = ''
    bands_report = []
    for bn in range(len(signalfinal[0])):
        band_result = signalfinal[:, bn]
        report = [np.sum(band_result == s) for s in range(CORRECT+1)]
        bands_report.append(report)

    bands_report = np.array(bands_report)
    final_report += '\n Signaling: how many events ' + \
                    'in each band signaled.\n'
    bands_header = '\n Band | '

    for signal in range(CORRECT+1):
        n_spaces = len(str(np.max(bands_report[:, signal])))-1
        bands_header += ' '*n_spaces+str(signal) + '   '

    final_report += bands_header + '\n'
    final_report += '-'*len(bands_header)

    for bn, report in enumerate(bands_report):
        final_report += f'\n {bn}{" "*(4-len(str(bn)))} |' + ' '
        for signal, value in enumerate(report):
            n_max = len(str(np.max(bands_report[:, signal])))
            n_spaces = n_max - len(str(value))
            final_report += ' '*n_spaces+str(value) + '   '

    print(final_report)

            