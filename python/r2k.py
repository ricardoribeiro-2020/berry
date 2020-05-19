# This program reads a set of wavefunctions of for different k and bands and translates that
# to another set for different points in real space, the functions become a function of k instead of r

# This is to include maths
import math
import cmath
import re
import numpy as np
from findiff import FinDiff, coefficients, Coefficient, Gradient

# This is to make operations in the shell
import sys
import time
# This are the subroutines and functions
#import interpolation
import load
import contatempo
# This to make parallel processing
import joblib

memory = joblib.Memory(cachedir='/tmp/example/',verbose=0)

@memory.cache
def indices3(i,j,l):
  index = str(i)+' '+str(j)+' '+str(l)
  return index

@memory.cache
def indices2(i,j):
  index = str(i)+' '+str(j)
  return index

nproc = -1                                     # Number of processors -1=all, -2=all-1,...

starttime = time.time()                        # Starts counting time

nbndmax = 8                                 # number of bands to be considered
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

occ = load.readoccupancies(wfcdirectory,nbnd)    # {'58 6': 0.0, '25 8': 0.0, ...

apontador,signaled = load.readsignaled(wfcdirectory,nbnd) # {'58 6': 8, '25 8': 8, '39 2': 3, ...

ban = load.readbandas(wfcdirectory,nbnd)             # kpoint,newband-> machine nr

eigenvalues = {}                                     # Eigenvalues of the ordered states !!! not sure
for ki in range(nks):
  for bandas in range(1,nbnd+1):
    indx = indices2(ki,bandas)
    if indx in apontador.keys():
      index = indices2(ki,apontador[indx])
      if index in enerBands.keys():
        eigenvalues[index] = enerBands[indx]             # {'25 8': 0.336136887984, '39 2': -0.455816584941, ...
      else:
        eigenvalues[index] = 0.0

#    print(indx,eigenvalues[index],enerBands[index])
#print(eigenvalues)

print(' Finished reading data')
sys.stdout.flush()
#sys.exit("Stop")

################################################## Finished reading data


kmatrix = np.zeros((numero_kx,numero_ky))
kmatriy = np.zeros((numero_kx,numero_ky))
kmatriz = np.zeros((numero_kx,numero_ky))
kmatx = np.full((numero_kx,numero_ky),0.+0j)
kmaty = np.full((numero_kx,numero_ky),0.+0j)
kmatz = np.full((numero_kx,numero_ky),0.+0j)
for j in range(numero_ky):
  for i in range(numero_kx):
    kp = kindex[indices3(i,j,0)]
    kmatrix[i,j] = kpointx[kp]
    kmatriy[i,j] = kpointy[kp]
    kmatriz[i,j] = kpointz[kp]

    kmatx[i,j] = -2j*math.pi*kmatrix[i,j]
    kmaty[i,j] = -2j*math.pi*kmatriy[i,j]
    kmatz[i,j] = -2j*math.pi*kmatriz[i,j]

################################################## Determine points in real space from first file
with open(wfcdirectory+"/k0000b0001.wfc",'r') as datfile:
  data = datfile.read().split('\n')
datfile.closed
del data[-1]
nr = len(data)                                       # Number of points in real space
r,x,y,z = {},{},{},{}                                # vector r and coordinates x,y,z
with open(wfcdirectory+"/rindex","w") as rindex:     # builds an index of r-points: not used
  for i in range(nr):
    tmp = data[i].split()
    del tmp[-1]
    del tmp[-1]
    del tmp[-1]
    rindex.write(str(i)+' '+tmp[0]+' '+tmp[1]+' '+tmp[2]+'\n')
    r[i] = tmp
    x[i] = float(tmp[0])
    y[i] = float(tmp[1])
    z[i] = float(tmp[2])
rindex.closed
################################################## 

grad = Gradient(h=[dk, dk],acc=3)            # Defines gradient function in 2D
################################################## 

bandasr = {}                           # Dictionary with wfc for all points and bands
bandasg = {}                           # Dictionary with wfc gradients for all points and bands

for banda in range(1,nbndmax+1):       # For each band
  wfct_k = {}                          # wfct_k is a dictionary that, for each k (i,j,l) gives a list 
  kp = -1                              #  with x y z psi^2 psir psii
  for l in range(numero_kz):           # Reads all wfc for all k-points and adds to dictionary wfct_k
    for j in range(numero_ky):
      for i in range(numero_kx):
        kp = kp + 1                    # k-point number, starts at zero
        indx = indices2(kp,banda)      # index for ban
        banda0 = ban[indx]             # Chooses the machine nr for reading the wfc
        index = indices3(i,j,l)        # ref. to k point being considered
        inex = indices2(kp,banda0)     # index for app

        if inex in signaled.keys():
          if signaled[inex] > 3 and banda0 < nbndmax+1:      # if its a signaled wfc, choose interpolated
            print('band '+str(indx)+' - mach. nr '+str(inex))
            wfct_k[index] = load.readwfc(wfcdirectory,kp,banda0,col='ab') 
          else:                          # else choose original
            wfct_k[index] = load.readwfc(wfcdirectory,kp,banda0,col='a')
        else:                          # else choose original
          wfct_k[index] = load.readwfc(wfcdirectory,kp,banda0,col='a')
#      {'4 6 0': ['0.0   0.0   0   24.642026286543164   4.964073557728891   0.0', ...

  print(' Finished reading wfc from files: ',str((time.time()-starttime)/60.),' min')
  sys.stdout.flush()

#  sys.exit("Stop")
  wfcpos = {}                          # Dictionary with wfc for all points
  wfcgra = {}                          # Dictionary with wfc gradients for all points

  for posi in range(nr):
    wfcpos[posi] = np.zeros((numero_kx,numero_ky),dtype=complex)   # Arrays represent the wfc in kspace
    wfcgra[posi] = np.zeros((numero_kx,numero_ky),dtype=complex)   # Arrays represent the grad(wfc) in kspace

  for l in range(numero_kz):
    for j in range(numero_ky):
      for i in range(numero_kx):       # for each kpoint
        index = indices3(i,j,l)
        for posi in range(nr):         # transform psi to u
          fase = cmath.exp(x[posi]*kmatx[i,j] + y[posi]*kmaty[i,j] + z[posi]*kmatz[i,j])
          wfcpos[posi][i,j] = fase*(float(wfct_k[index][posi].split()[4]) + 1j*float(wfct_k[index][posi].split()[5]))

  for posi in range(nr):
    wfcgra[posi] = grad(wfcpos[posi])                      # Complex gradient


  bandasr[banda] = wfcpos
  bandasg[banda] = wfcgra
  print(' Finished band ',str(banda),'   ',str((time.time()-starttime)/60.),' min')
  sys.stdout.flush()

  joblib.dump(wfcpos,'./wfcpos'+str(banda)+'.gz', compress=3)
  joblib.dump(wfcgra,'./wfcgra'+str(banda)+'.gz', compress=3)


sys.stdout.flush()

# Cleaning
memory.clear(warn=False)

endtime = time.time()
print(contatempo.tempo(starttime,endtime))



#sys.exit("Stop")









