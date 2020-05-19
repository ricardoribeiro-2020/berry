# This program reads a set of wavefunctions of for different k and bands and translates that
# to another set for different points in real space, the functions become a function of k instead of r

# This is to include maths
import math
import re
import numpy as np
from findiff import FinDiff, coefficients, Coefficient, Gradient

# This is to make operations in the shell
import sys
import time
# This are the subroutines and functions
import load
import contatempo
# This to make parallel processing
import joblib

#memory = joblib.Memory(cachedir='/tmp/example/',verbose=0)

#@memory.cache
def indices3(i,j,l):
  index = str(i)+' '+str(j)+' '+str(l)
  return index

#@memory.cache
def indices2(i,j):
  index = str(i)+' '+str(j)
  return index

if len(sys.argv)!=3:
  print(' ERROR in number of arguments. Has to have two integers.')
  sys.exit("Stop")

bandwfc = int(sys.argv[1])
gradwfc = int(sys.argv[2])
#print(str(bandwfc),str(gradwfc))

starttime = time.time()                   # Starts counting time
wfcdirectory = 'wfc'

################################################ Read the k-points, eigenvalues, bands
print(' Start reading data')

tmp = load.readkindex(wfcdirectory)
numero_kx = tmp[0]            # Number of k-points in each direction
numero_ky = tmp[1]
numero_kz = tmp[2]
kindex = tmp[3]               # {'7 7 0': 63, '7 4 0': 39, '4 5 0': 44,...
nks = tmp[4]                  # number of k-points in the calculation: nks
kpointx = tmp[5]              # {0: 0.0, 1: 0.01, 2: 0.02, ...
kpointy = tmp[6]
kpointz = tmp[7]
dk = tmp[8]                   # Defines the step for gradient calculation dk

tmp = load.readenergias(wfcdirectory)
nbnd = tmp[0]                 # Number of bands in the DFT calculation: nbnd
enerBands = tmp[1]            # {'58 6': 0.28961845029, '25 8': 0.336136887984, ...
eigenvalues = tmp[3]

occ = load.readoccupancies(wfcdirectory,nbnd)    # {'58 6': 0.0, '25 8': 0.0, ...

apontador = load.readapontador(wfcdirectory,nbnd) # {'58 6': 8, '25 8': 8, '39 2': 3, ...

ban = load.readbandas(wfcdirectory,nbnd)             # kpoint,newband-> old band

print(' Start reading dump files')
wfcpos = joblib.load('./wfcpos'+str(bandwfc)+'.gz')
wfcgra = joblib.load('./wfcgra'+str(gradwfc)+'.gz')

print(' Finished reading data')
sys.stdout.flush()

################################################## Finished reading data

nr = len(wfcpos)    # Number of points in real space that constitues the wavefunction

################################################## Calculation of the Berry connection

berryConnection = 1j*wfcpos[0].conj()*wfcgra[0]

for posi in range(1,nr):
  berryConnection += 1j*wfcpos[posi].conj()*wfcgra[posi] 
# we are assuming that normalization is \sum |\psi|^2 = 1
# if not, needs division by nr

print(' Finished calculating Berry connection for index '+str(bandwfc)+'  '+str(gradwfc)+'.\
                                  \n Saving results to file')
sys.stdout.flush()

filename = wfcdirectory+'/berryCon'+str(bandwfc)+'-'+str(gradwfc)
#print(filename)
# output units of Berry connection are bohr

with open(filename,'w') as bc:  # Saving Berry connection file
  bc.write(str(berryConnection))
bc.closed

joblib.dump(berryConnection,filename+'.gz', compress=3)


# Cleaning
#memory.clear(warn=False)

endtime = time.time()
print(contatempo.tempo(starttime,endtime))



#sys.exit("Stop")









