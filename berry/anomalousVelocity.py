"""
 This program calculates the anomalous velocity from the Berry curvature
"""

import sys
import time
import logging

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
except:
    pass

def run_anomalous_velocity(logger_name: str = "anomalousVelocity", logger_level: int = logging.INFO, flush: bool = True):
    looger = log(logger_name, "ANOMALOUS VELOCITY", level=logger_level, flush=flush)
    return

if __name__ == "__main__":
    pass # Add run_anomalous_velocity() here (name can be changed)

