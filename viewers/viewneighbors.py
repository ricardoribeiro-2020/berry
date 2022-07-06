""" This program lists the neighbors of each k-point
"""

import numpy as np
import loaddata as d

if __name__ == "__main__":

    for nk in range(d.nks):
        print(
            nk,
            d.neighbors[nk, 0],
            d.neighbors[nk, 1],
            d.neighbors[nk, 2],
            d.neighbors[nk, 3],
        )
