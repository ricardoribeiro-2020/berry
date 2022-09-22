"""
 This program calculates the anomalous velocity from the Berry curvature
"""

import sys
import time

import numpy as np

# This are the subroutines and functions
from contatempo import tempo, inter_time
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("ANOMALOUS VELOCITY", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time











    ##################################################################################r
    # Finished
    footer(tempo(STARTTIME, time.time()))

