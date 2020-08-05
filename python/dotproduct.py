###################################################################################
# This program calculates the dot product of the wfc Bloch factor with their neighbors
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time
import subprocess

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d

header('DOTPRODUCT',time.asctime())

starttime = time.time()                         # Starts counting time

# Reading data needed for the run
berrypath = str(d.berrypath)
print(' Path to BERRY files:',berrypath)

wfcdirectory = str(d.wfcdirectory)
print(' Directory where the wfc are:',wfcdirectory)
nkx = d.nkx
nky = d.nky
nkz = d.nkz
print(' Number of k-points in each direction:',nkx,nky,nkz)
nks = d.nks
print(' Total number of k-points:',nks)

nr1 = d.nr1
nr2 = d.nr2
nr3 = d.nr3
print(' Number of points in each direction:',nr1,nr2,nr3)
nr = d.nr
print(' Total number of points in real space:',nr)
npr = d.npr
print(' Number of processors to use',npr)

nbnd = d.nbnd
print(' Number of bands:',nbnd)
print()

phase = d.phase
print(' Phases loaded')
#print(phase[10000,10])

neighbors = d.neighbors
print(' Neighbors loaded')

# Finished reading data needed for the run

with open('tmp','w') as tmp:             # Creates temporary file to store the number of points in r-space
  tmp.write(str(nr))
tmp.closed
print()
os.system('rm -f dp.dat dpc.dat')
os.system('touch dp.dat dpc.dat')
sys.stdout.flush()

for nk in range(nks):                    # runs through all k-points
  for j in range(4):                     # runs through all neighbors
    if neighbors[nk,j] != -1:            # exclude invalid neighbors
#      print(nk,j,neighbors[nk,j])
      comando = "&input  \
         nk = "+str(nk)  \
  +",  nbnd = "+str(nbnd)\
  +",    nr = "+str(nr) \
  +",   npr = "+str(npr) \
  +", wfcdirectory = '"+wfcdirectory+"'"\
  +", neighbor = "+str(neighbors[nk,j])+",   "
      for i in range(3):
        comando += "phase("+str(i)+",0)=("+str(np.real(phase[i,nk]))+","+str(np.imag(phase[i,nk]))+"),"
        comando += "phase("+str(i)+",1)=("+str(np.real(phase[i,neighbors[nk,j]]))+","+str(np.imag(phase[i,neighbors[nk,j]]))+"),"
      comando += " / "                   # prepares command to send to f90 program connections.x

      print("Calculating   nk = "+str(nk)+"  neighbor = "+str(neighbors[nk,j]))
#      print('echo "'+comando+'"|'+berrypath+'bin/connections.x')
      sys.stdout.flush()

      os.system('echo "'+comando+'"|'+berrypath+'bin/connections.x')      # Runs f90 program connections.x
#      sys.exit("Stop")

#os.system('rm tmp')
# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))




