"""This module computes the band classification
using the algorithm in clustering_libs.py
"""

import sys
import getopt
import os
import numpy as np
import contatempo
import time
import loaddata as d
from headerfooter import header, footer
from clustering_libs import MATERIAL

N_PROCESS = os.cpu_count()
TOL = 0.95

if __name__ == '__main__':
    if len(sys.argv) > 1:
        opts, args = getopt.getopt(sys.argv[1:], "n:t:")
        if len(opts) > 0:
            for opt, arg in opts:
                if opt == '-n':
                    N_PROCESS = int(arg)
                if opt == '-t':
                    TOL = float(arg)
    else:
        args = []

    header("Band Clustering ", d.version, time.asctime())

    STARTTIME = time.time()

    try:
        bands = [0, int(args)] if len(args) == 1 else None
        bands = [int(v) for v in args] if len(args) == 2 else [0, d.nbnd-1]
    except:
        bands = [0, d.nbnd-1]
        print(f'Warning: The arguments given do not correspond to the expected.\
             \n          The program will use the default settings. BANDS: 0-{d.nbnd-1}')

    min_band, max_band = bands
    n_bands = max_band-min_band+1

    print(f'     Min band: {min_band}    Max band: {max_band}')
    print(f'     Tolerance: {TOL}')
    print(f'     Number of CPUs: {N_PROCESS}\n')

    print("     Unique reference of run:", d.refname)
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Neighbors loaded")
    print("     Eigenvalues loaded")

    connections = np.load("dp.npy")
    print("     Modulus of direct product loaded")

    print()
    print("     Finished reading data")
    print()

    material = MATERIAL(d.nkx, d.nky, d.nbnd, d.nks, d.eigenvalues,
                        connections, d.neighbors, n_process=N_PROCESS)

    print('\n  Calculating Vectors', end=': ')
    init_time = time.time()
    material.make_vectors(min_band=min_band, max_band=max_band)
    print(f'{contatempo.tempo(init_time, time.time())}')

    print('  Calculating Connections Matrix', end=': ')
    init_time = time.time()
    material.make_connections(tol=TOL)
    print(f'{contatempo.tempo(init_time, time.time())}')

    material.clear_temp()

    print('  Calculating Componets Matrix', end=': ')
    init_time = time.time()
    labels = material.get_components()
    print(f'{contatempo.tempo(init_time, time.time())}')

    if not os.path.exists('output/'):
        os.mkdir('output/')

    with open('output/VECTORS.npy', 'wb') as f:
        np.save(f, material.nkx)
        np.save(f, material.nky)
        np.save(f, material.vectors)
        np.save(f, labels)

    print('Clustering Done\n')

    print('    Computing bandsfinal', end=': ')
    init_time = time.time()
    material.obtain_output()
    print(f'{contatempo.tempo(init_time, time.time())}')

    with open('output/bandsfinal.npy', 'wb') as f:
        np.save(f, material.bands_final)

    with open('output/signalfinal.npy', 'wb') as f:
        np.save(f, material.signal_final)

    with open('output/degeneratefinal.npy', 'wb') as f:
        np.save(f, material.degenerate_final)

    material.print_report()

    with open('output/final.report', 'w') as f:
        f.write(material.final_report)

    ENDTIME = time.time()
    footer(contatempo.tempo(STARTTIME, ENDTIME))
