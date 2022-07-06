""" This program writes to output the eigenvalues of a specified band
"""

import sys
import numpy as np
import loaddata as d
from write_k_points import float_numbers

if __name__ == "__main__":

    if len(sys.argv) == 2:
        band = int(sys.argv[1])
        precision = 2
    elif len(sys.argv) == 3:
        band = int(sys.argv[1])
        precision = int(sys.argv[2])
    else:
        band = 0
        precision = 2

    #    print(d.eigenvalues.shape)

    float_numbers(d.nkx, d.nky, d.eigenvalues[:, band], precision)
