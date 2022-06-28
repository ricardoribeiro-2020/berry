""" This program lists the occupancies of each band in k-space
"""

import numpy as np
import loaddata as d

if __name__ == "__main__":

    for nk in range(d.nks):
        print(nk, d.occupations[nk, :])
