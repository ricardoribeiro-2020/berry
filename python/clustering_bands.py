"""This module computes the band classification
using the algorithm in clustering_libs.py

python clustering_bands.py

Some flags are permitted.
    -n: set the number of CPUs
    -t: set the tolerance

Finally, It is possible to choose the group of bands
to be solved given as argument the maximum band
or the interval of minimum and maximum band.
The default uses all bands.
"""

import sys
import getopt
import os
import numpy as np
import contatempo
import time
import loaddata as d
from clustering_libs import MATERIAL
from log_libs import log

N_PROCESS = d.npr
TOL = 0.95

if __name__ == '__main__':
    if len(sys.argv) > 1:
        opts, args = getopt.getopt(sys.argv[1:], "n:t:", ["np ="])
        if len(opts) > 0:
            for opt, arg in opts:
                if opt in ['-n', '--np', '--np ']:
                    N_PROCESS = int(arg)
                if opt == '-t':
                    TOL = float(arg)
    else:
        args = []

    LOG = log('clustering', 'Band Clustering', d.version)
    LOG.header()

    STARTTIME = time.time()

    try:
        bands = [0, int(args[0])] if len(args) == 1 else \
                [int(v) for v in args] if len(args) == 2 else [0, d.nbnd-1]
    except Exception:
        bands = [0, d.nbnd-1]
        warning_msg = 'WARNING: The arguments given do not correspond'\
                      + ' to the expected.'\
                      + '\n\tThe program will use the default settings.'\
                      + f'BANDS: 0-{d.nbnd-1}'
        print(warning_msg)
    
    if not os.path.exists('output/'):
        os.mkdir('output/')

    min_band, max_band = bands
    n_bands = max_band-min_band+1

    LOG.info(f'     Min band: {min_band}    Max band: {max_band}')
    LOG.info(f'     Tolerance: {TOL}')
    LOG.info(f'     Number of CPUs: {N_PROCESS}\n')

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

    print()
    LOG.info("     Finished reading data")
    print()

    material = MATERIAL(d.nkx, d.nky, d.nbnd, d.nks, d.eigenvalues,
                        connections, d.neighbors, n_process=N_PROCESS)

    LOG.info('\n  Calculating Vectors')
    init_time = time.time()
    material.make_vectors(min_band=min_band, max_band=max_band)
    LOG.info(f'{contatempo.tempo(init_time, time.time())}')

    LOG.info('  Calculating Connections')
    init_time = time.time()
    material.make_connections(tol=TOL)
    LOG.info(f'{contatempo.tempo(init_time, time.time())}')

    LOG.info('  Solving problem')
    init_time = time.time()
    labels = material.solve()
    LOG.info(f'{contatempo.tempo(init_time, time.time())}')

    LOG.info('Clustering Done')

    with open('output/final.report', 'w') as f:
        f.write(material.final_report)

    with open('output/bandsfinal.npy', 'wb') as f:
        np.save(f, material.bands_final)

    with open('output/signalfinal.npy', 'wb') as f:
        np.save(f, material.signal_final)

    with open('output/degeneratefinal.npy', 'wb') as f:
        np.save(f, material.degenerate_final)

    LOG.footer()

