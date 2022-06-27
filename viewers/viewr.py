""" This program lists the coordinates of each point in r-space
"""

import numpy as np
import loaddata as d

if __name__ == "__main__":

    for i in range(d.nr):
        print(i, d.r[i, 0], d.r[i, 1], d.r[i, 2])
