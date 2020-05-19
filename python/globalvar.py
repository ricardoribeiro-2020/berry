# This program initializes all global variables

# This is to include maths
import math
import numpy as np
# This is to make operations in the shell
import sys
import os
import time
# This to make parallel processing
import joblib
# This are the subroutines and functions
import contatempo
import load
from auxiliary import indices2, indices3

starttime = time.time()                         # Starts counting time

wfcdirectory = 'wfc'                            # Directory to store the final wfc
warnings = '\n * Warnings *\n'                    # String to include in the end of output
print(' Start reading data')

tmp = load.readkindex(wfcdirectory)     # Loading k-point mesh
numero_kx = tmp[0]            # Number of k-points in each direction
numero_ky = tmp[1]
numero_kz = tmp[2]
kindex = tmp[3]               # {'7 7 0': 63, '7 4 0': 39, '4 5 0': 44,...
nks = tmp[4]                  # number of k-points in the calculation: nks
kpointx = tmp[5]              # {0: 0.0, 1: 0.01, 2: 0.02, ...
kpointy = tmp[6]
kpointz = tmp[7]
dk = tmp[8]                   # Defines the step for gradient calculation dk

tmp = load.readenergias1(wfcdirectory)  # Loading eigenvalues
nbnd = tmp[0]                 # Number of bands in the DFT calculation: nbnd
enerBands = tmp[1]            # {'58 6': 0.28961845029, '25 8': 0.336136887984, ...

occ = load.readoccupancies(wfcdirectory,nbnd)   # {'58 6': 0.0, '25 8': 0.0, ...

bandstolerance = 16                             # Number of bands considered valid (starting in the lowest)
tolerancia = 0.08                               # Value to consider other bands close (Ha)
orto = 0.2                                      # tolerance for ortogonality (1.0 +/- orto)

print(' Number of bands in the nscf calculation ',str(nbnd))
print(' Number of bands considered valid (starting in the lowest) ',str(bandstolerance))
print(' Tolerance of orthoganality: ',str(orto))
print(' Maximum difference in energy to consider a possible band ',str(tolerancia),' Ry')
print(' Number of points in the mesh ',str(numero_kx),str(numero_ky),str(numero_kz))
print(' Total number of k-points ',str(nks))
print(' Diferential dk: ',str(dk))
print()
print(' Finished reading data      '+str((time.time()-starttime)/60.),' min')
sys.stdout.flush()
################################################## Finished reading data


dege = list(range(1, nbnd+1))                   # defines list of bands to compare: original order

