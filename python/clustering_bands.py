"""
This module computes the band classification
using the algorithm in clustering_libs.py

python clustering_bands.py

Some flags are permitted.
    -np: set the number of CPUs
    -t: set the tolerance
    -mb: set the minimum band

Finally, It is possible to choose the group of bands
to be solved given as argument the maximum band
or the interval of minimum and maximum band.
The default uses all bands.
"""
import os
import time
import numpy as np
import loaddata as d

from log_libs import log
from cli import clustering_cli

LOG = log('clustering', 'Band Clustering', d.version)

from clustering_libs import MATERIAL

if __name__ == '__main__':
    args = clustering_cli()
    
    LOG.header()
    STARTTIME = time.time()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    OUTPUT_PATH = ''
    NPR = args['NPR']
    TOL = args["TOL"]
    max_band = args['MAX_BAND'] if args['MAX_BAND'] != -1 else d.nbnd-1
    bands = [args['MIN_BAND'], max_band]

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
        LOG.warning(f'The {OUTPUT_PATH} was created.')

    min_band, max_band = bands
    n_bands = max_band-min_band+1

    LOG.info(f'     Min band: {min_band}    Max band: {max_band}')
    LOG.info(f'     Tolerance: {TOL}')
    LOG.info(f'     Number of CPUs: {NPR}\n')

    LOG.info(f"     Unique reference of run:{d.refname}")
    LOG.info(f"     Directory where the wfc are:{d.wfcdirectory}")
    LOG.info(f"     Number of k-points in each direction:{d.nkx}, {d.nky}, {d.nkz}")
    LOG.info(f"     Total number of k-points:{d.nks}")
    LOG.info(f"     Number of bands:{d.nbnd}")
    LOG.info()
    LOG.info("     Neighbors loaded")
    LOG.info("     Eigenvalues loaded")

    connections = np.load("dp.npy")
    LOG.info("     Modulus of direct product loaded")

    LOG.info()
    LOG.info("     Finished reading data")
    LOG.info()

    ###########################################################################
    # 3. CLUSTERING ALGORITHM
    ########################################################################### 

    material = MATERIAL(d.nkx, d.nky, d.nbnd, d.nks, d.eigenvalues,
                        connections, d.neighbors, n_process=NPR)

    LOG.info('\n  Calculating Vectors')
    material.make_vectors(min_band=min_band, max_band=max_band)

    LOG.info('  Calculating Connections')
    material.make_connections(tol=TOL)

    LOG.info('  Solving problem')
    material.solve()

    LOG.info('Clustering Done')

    with open(f'{OUTPUT_PATH}final.report', 'w') as f:
        f.write(material.final_report)

    with open(f'{OUTPUT_PATH}bandsfinal.npy', 'wb') as f:
        np.save(f, material.bands_final)

    with open(f'{OUTPUT_PATH}signalfinal.npy', 'wb') as f:
        np.save(f, material.signal_final)

    with open(f'{OUTPUT_PATH}correct_signalfinal.npy', 'wb') as f:
        np.save(f, material.correct_signalfinal)

    with open(f'{OUTPUT_PATH}degeneratefinal.npy', 'wb') as f:
        np.save(f, material.degenerate_final)

    with open(f'{OUTPUT_PATH}final_score.npy', 'wb') as f:
        np.save(f, material.final_score)

    LOG.footer()

