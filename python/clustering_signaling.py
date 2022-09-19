import numpy as np
import loaddata as d
from clustering_libs import evaluate_point
from log_libs import log
import time

OLD_CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3

CORRECT = 4
OTHER = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

if __name__ == "__main__":
    
    LOG = log('signaling', 'Band Clustering Signaling', d.version)
    LOG.header()

    # Reading data needed for the run
    berrypath = d.berrypath

    LOG.info(f"     Unique reference of run:{d.refname}")
    LOG.info(f"     Directory where the wfc are:{d.wfcdirectory}")
    LOG.info(f"     Number of k-points in each direction:{d.nkx}, {d.nky}, {d.nkz}")
    LOG.info(f"     Total number of k-points:{d.nks}")
    LOG.info(f"     Number of bands:{d.nbnd}")
    print()
    LOG.info("     Neighbors loaded")
    LOG.info("     Eigenvalues loaded")

    connections = np.load("dp.npy")
    LOG.info("     Modulus of direct product loaded")

    LOG.info("     Reading files bandsfinal.npy and signalfinal.npy")
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()

    print()
    LOG.info("     Finished reading data")
    print()

    My, Mx = np.meshgrid(np.arange(d.nky), np.arange(d.nkx))
    k_matrix = My*d.nkx+Mx
    counts = np.arange(d.nks)
    k_index = np.stack([counts % d.nkx, counts//d.nkx], axis=1)


    correct_signalfinal = np.copy(signalfinal)
    correct_signalfinal[signalfinal == OLD_CORRECT] = CORRECT

    ks_pC, bnds_pC = np.where(signalfinal == POTENTIAL_CORRECT)
    ks_pM, bnds_pM = np.where(signalfinal == POTENTIAL_MISTAKE)

    ks = np.concatenate((ks_pC, ks_pM))
    bnds = np.concatenate((bnds_pC, bnds_pM))

    for k, bn in zip(ks, bnds):
        correct_signalfinal[k, bn], scores = evaluate_point(k, bn, k_index, k_matrix, signalfinal, bandsfinal, d.eigenvalues)
        LOG.debug(f'K point: {k} Band: {bn}')
        LOG.debug(f'    New Signal: {correct_signalfinal[k, bn]}')
        LOG.debug(f'    Directions: {scores}')
    
    final_report = ''
    bands_report = []
    for bn in range(len(correct_signalfinal[0])):
        band_result = correct_signalfinal[:, bn]
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

            