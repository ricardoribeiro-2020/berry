###################################################################################
# This program calculates the dot product of the wfc Bloch factor with their neighbors
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d
import connections

header('DOTPRODUCT',time.asctime())

starttime = time.time()                         # Starts counting time

# Reading data needed for the run

wfcdirectory = str(d.wfcdirectory)
print(' Directory where the wfc are:',wfcdirectory)
nks = d.nks
print(' Total number of k-points:',nks)

nr = d.nr
print(' Total number of points in real space:',nr)
npr = d.npr
print(' Number of processors to use',npr)

nbnd = d.nbnd
print(' Number of bands:',nbnd)
print()

phase = d.phase
print(' Phases loaded')
#print(phase[10000,10]) # phase[nr,nks]

neighbors = d.neighbors
print(' Neighbors loaded')

# Finished reading data needed for the run
print()
os.system('rm -f dp.dat dpc.dat')
os.system('touch dp.dat dpc.dat')
sys.stdout.flush()

for nk in range(nks):                    # runs through all k-points
  for j in range(4):                     # runs through all neighbors
    neighbor = neighbors[nk,j]
    if neighbor != -1:            # exclude invalid neighbors
#      print(nk,j,neighbors[nk,j])

      phases = np.stack((phase[:,nk],phase[:,neighbor]),axis=1)

      print("Calculating   nk = "+str(nk)+"  neighbor = "+str(neighbor))
      sys.stdout.flush()
      connections.connect(nk,nbnd,npr,neighbor,wfcdirectory,phases,nr)
#    sys.exit("Stop")

# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))




