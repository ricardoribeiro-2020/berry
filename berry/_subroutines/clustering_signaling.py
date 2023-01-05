import logging

import numpy as np

from berry import log, __version__
from clustering_libs import evaluate_point

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass

OLD_CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3

CORRECT = 4
OTHER = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

if __name__ == "__main__":
    
    LOG = log('signaling', 'Band Clustering Signaling', __version__, level=logging.DEBUG)
    LOG.header()

    # Reading data needed for the run
    berrypath = m.berrypath

    LOG.info(f"\tUnique reference of run:{m.refname}")
    LOG.info(f"\tDirectory where the wfc are:{m.wfcdirectory}")
    LOG.info(f"\tNumber of k-points in each direction:{m.nkx}, {m.nky}, {m.nkz}")
    LOG.info(f"\tTotal number of k-points:{m.nks}")
    LOG.info(f"\tNumber of bands:{m.nbnd}")
    print()
    LOG.info("\tNeighbors loaded")
    LOG.info("\tEigenvalues loaded")

    connections = np.load("dp.npy")
    LOG.info("\tModulus of direct product loaded")

    LOG.info("\tReading files bandsfinal.npy and signalfinal.npy")
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()

    print()
    LOG.info("\tFinished reading data")
    print()

    My, Mx = np.meshgrid(np.arange(m.nky), np.arange(m.nkx))
    k_matrix = My*m.nkx+Mx
    counts = np.arange(m.nks)
    k_index = np.stack([counts % m.nkx, counts//m.nkx], axis=1)


    correct_signalfinal = np.copy(signalfinal)
    correct_signalfinal[signalfinal == OLD_CORRECT] = CORRECT

    ks_pC, bnds_pC = np.where(signalfinal == POTENTIAL_CORRECT)
    ks_pM, bnds_pM = np.where(signalfinal == POTENTIAL_MISTAKE)

    ks = np.concatenate((ks_pC, ks_pM))
    bnds = np.concatenate((bnds_pC, bnds_pM))

    error_directions = []
    directions = []

    for k, bn in zip(ks, bnds):
        signal, scores = evaluate_point(k, bn, k_index, k_matrix, signalfinal, bandsfinal, d.eigenvalues)
        correct_signalfinal[k, bn] = signal
        if signal < CORRECT:
            error_directions.append([k, bn])
            directions.append(scores)
        LOG.debug(f'\tK point: {k} Band: {bn}')
        LOG.debug(f'\tNew Signal: {signal}')
        LOG.debug(f'\tDirections: {scores}')
    
    error_directions = np.array(error_directions)
    directions = np.array(directions)

    bands_signaling = np.zeros((m.nbnd, 4, *k_matrix.shape), int)
    k_index = k_index[error_directions[:, 0]]
    ik, jk = k_index[:, 0], k_index[:, 1]
    bnds = error_directions[:, 1]
    repeat_scores = np.sum(directions == 0, axis=1)
    bnds = np.repeat(bnds, repeat_scores)
    ik = np.repeat(ik, repeat_scores)
    jk = np.repeat(jk, repeat_scores)
    scores = np.where(directions == 0)[1]
    bands_signaling[bnds, scores, ik, jk] = 1
    for bn, band in enumerate(bands_signaling):
        with open(f'signaling/bn_{bn}_signaling.npy', 'wb') as f:
            np.save(f, band)



    final_report = ''
    bands_report = []
    for bn in range(len(correct_signalfinal[0])):
        band_result = correct_signalfinal[:, bn]
        report = [np.sum(band_result == s) for s in range(CORRECT+1)]
        bands_report.append(report)

    bands_report = np.array(bands_report)
    final_report += ' Signaling: how many events ' + \
                    'in each band d.\n'
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

    LOG.info(final_report)