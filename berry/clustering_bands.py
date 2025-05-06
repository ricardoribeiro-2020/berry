"""
-----------------------------------------------------------------------------------------------------------------------
Project Title: Cluster Solver for Band Structure Analysis
-----------------------------------------------------------------------------------------------------------------------

Last Modified: [2025-05-06]

Description:
------------
This software package is designed to solve for clusters of components in band structure analysis. It provides an 
iterative process to group components across multiple bands based on their mesh compatibility, energy levels, and 
intersections. The clustering process is sensitive to degeneracy, and the algorithm ensures that degenerate points 
are handled accordingly.

The core of the algorithm works by processing each band, grouping compatible components together, and returning the 
final set of clusters. The solution can be executed either sequentially or in parallel, providing flexibility for 
large-scale problems.


Usage:
------
This package is used for clustering components in band structure analysis and can be invoked as shown bellow

        usage: berry cluster [-h] [-mb [0-25]] [-np [1-112]] [-t [0.0-1.0]] [-flush] [-o file_path] [-v] [Mb 0-25]

        Classifies the eigenstates in bands.

        positional arguments:
        Mb (0-25)     Maximum band to consider.

        optional arguments:
        -h, --help    show this help message and exit
        -mb [0-25]    Minimum band to consider (default: 0).
        -np [1-112]   Number of processes to use (default: 1).
        -t [0.0-1.0]  Tolerance used for graph construction (default: 0.95).
        -flush        Flushes output into stdout.
        -o file_path  Name of output log file. Extension will be .log regardless of user input.
        -v            Increases output verbosity.

For detailed usage instructions, please refer to the documentation.
-----------------------------------------------------------------------------------------------------------------------
"""


import os
import sys
import logging
import time

import numpy as np


from berry import log

from berry._subroutines.clustering_algorithm import Material

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def run_clustering(max_band: int, min_band: int = 0, tol: float = 0.99, npr: int = 1, logger_name: str = "cluster", logger_level: int = logging.INFO, flush: bool = False, verbose=False):
    logger = log(logger_name, "CLUSTER", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    OUTPUT_PATH = ''
    max_band = max_band if max_band != -1 else m.final_band
    min_band = m.initial_band

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    if OUTPUT_PATH != "" and not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
        logger.warning(f'The {OUTPUT_PATH} was created.')

    logger.info(f'\tMin band: {min_band}    Max band: {max_band}')
    logger.info(f'\tTolerance: {tol}')
    logger.info(f'\tNumber of CPUs: {npr}\n')

    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx}, {m.nky}, {m.nkz}")
    logger.info(f"\tTotal number of k-points: {m.nks}")
    logger.info(f"\tNumber of bands: {m.number_of_bands}\n")
    logger.info("\tNeighbors loaded")
    logger.info("\tEigenvalues loaded")

    connections = np.abs(np.load("data/dpc.npy").real)
    logger.info("\tModulus of direct product loaded\n")

    logger.info("\tFinished reading data\n")

    ###########################################################################
    # 3. CLUSTERING ALGORITHM
    ########################################################################### 

    delta_k = m.step


    material = Material(m.dimensions, [m.nkx, m.nky, m.nkz], m.nbnd, m.nks, d.eigenvalues,
                        connections, d.neighbors, logger, min_band, max_band, delta_k, n_process=npr, verbose=verbose)
    
    material.solver(tol)
    logger.info("\n")
    material.report()
    logger.info(material.final_report)

    '''
    ------------------------------------------------------------------------
    Saving the results
    ------------------------------------------------------------------------
    '''

    logger.info('\n\tClustering Done')

    m.data_dir = 'data/'

    with open(os.path.join(m.data_dir, 'bandsfinal.npy'), 'wb') as f:
        np.save(f, material.bands_final)

    with open(os.path.join(m.data_dir, 'signalfinal.npy'), 'wb') as f:
        np.save(f, material.signal_final)

    with open(os.path.join(m.data_dir, 'degeneratefinal.npy'), 'wb') as f:
        np.save(f, material.degenerate_final)

    sys.stdout.write('\n')
    logger.footer()

if __name__ == "__main__":
    run_clustering(9, log("clustering", "CLUSTERING", "version", logging.DEBUG))
