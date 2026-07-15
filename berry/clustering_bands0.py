# May 19, 2024
import os
import sys
import logging

import numpy as np

from berry import log
from berry._subroutines.clustering_libs import MATERIAL

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except ImportError:
    pass


def run_clustering(max_band: int, tol: float = 0.80, alpha : float = 1.0, step : float = 0.5, npr: int = 1, logger_name: str = "cluster", logger_level: int = logging.INFO, flush: bool = False):
    if "m" not in globals() or "d" not in globals():
        raise SystemExit("berry data not found -- run inside a preprocessed working directory "
                         "(the data/ metadata could not be imported).")

    logger = log(logger_name, "CLUSTER", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    max_band = max_band if max_band != -1 else m.final_band
    # The minimum band is fixed by the preprocessing cutoff: every band-indexed
    # file downstream of preprocessing (wfc files, dp.npy) starts at
    # m.initial_band, so the clustering cannot start at any other band.
    min_band = m.initial_band

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
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

    connections = np.load(os.path.join(m.data_dir, "dp.npy"))
    logger.info("\tModulus of direct product loaded\n")

    logger.info("\tFinished reading data\n")

    ###########################################################################
    # 3. CLUSTERING ALGORITHM
    ########################################################################### 

    material = MATERIAL(m.dimensions, [m.nkx, m.nky, m.nkz], m.nbnd, m.nks, d.eigenvalues,
                        connections, d.neighbors, logger, min_band,
                        max_band=max_band, n_process=npr)

    logger.info('\tCalculating Vectors')
    material.make_vectors()

    logger.info('\n\tCalculating Connections')
    material.make_connections(tol=tol)

    logger.info('\tSolving problem')
    material.solve(step=step, alpha=alpha)

    logger.info('\n\tClustering Done')

    with open(os.path.join(m.data_dir, 'final.report'), 'w') as f:
        f.write(material.final_report)

    with open(os.path.join(m.data_dir, 'bandsfinal.npy'), 'wb') as f:
        np.save(f, material.bands_final)

    with open(os.path.join(m.data_dir, 'signalfinal.npy'), 'wb') as f:
        np.save(f, material.signal_final)

    with open(os.path.join(m.data_dir, 'correct_signalfinal.npy'), 'wb') as f:
        np.save(f, material.correct_signalfinal)

    with open(os.path.join(m.data_dir, 'degeneratefinal.npy'), 'wb') as f:
        np.save(f, material.degenerate_final)

    with open(os.path.join(m.data_dir, 'final_score.npy'), 'wb') as f:
        np.save(f, material.final_score)
    
    with open(os.path.join(m.data_dir, 'completed_bands.npy'), 'wb') as f:
        np.save(f, material.completed_bands)

    # (nks, nslots) mask of the dp-broken seam k-points (non-FAIL bands) that the
    # Berry-geometry pass interpolates over -- exactly the report's signalled
    # dpBrk points, nothing more. See MATERIAL.dp_interp_mask / berry_geometry.
    with open(os.path.join(m.data_dir, 'dp_interp_mask.npy'), 'wb') as f:
        np.save(f, material.dp_interp_mask())

    sys.stdout.write('\n')
    logger.footer()
