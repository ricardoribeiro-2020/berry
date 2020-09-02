###################################################################################
# This program finds the problematic cases and makes an interpolation of 
# the wavefunctions or, if not necessary, take decisions about what wfc to use
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time

# These are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d
import interpolat

header('DOTPRODUCT',time.asctime())

starttime = time.time()                         # Starts counting time

if len(sys.argv)!=2:
  print(' ERROR in number of arguments. You probably want to give the last band to be considered.')
  sys.exit("Stop")

lastband = int(sys.argv[1])

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

nr = d.nr
print(' Total number of points in real space:',nr)
npr = d.npr
print(' Number of processors to use',npr)

nbnd = d.nbnd
print(' Number of bands:',nbnd)
print()

neighbors = d.neighbors
print(' Neighbors loaded')

eigenvalues = d.eigenvalues
print(' Eigenvlaues loaded')

dp = np.loadtxt('dp.dat')
print(' Modulus of direct product loaded')

connections = np.full((nks,4,nbnd,nbnd),-1.0)
for i in range(dp.shape[0]):
  nk1 = int(dp[i,0])
  nk2 = int(dp[i,1])
  b1  = int(dp[i,2]) - 1           # converts bands range from 1->nbdn to 0->nbdn-1
  b2 = int(dp[i,3]) - 1
  dotp = dp[i,4]
  for j in range(4):
    if nk2 == neighbors[nk1,j]:
      connections[nk1,j,b1,b2] = dotp

print(' Reading files bandsfinal.npy and signalfinal.npy')
with open('bandsfinal.npy', 'rb') as f:
  bandsfinal = np.load(f)
f.closed
with open('signalfinal.npy', 'rb') as f:
  signalfinal = np.load(f)
f.closed



###################################################################################

print()
print('**********************')
print(' Problems not solved')
kpproblem,bnproblem = np.where(signalfinal == -1)
print(kpproblem)
print(bnproblem)
print(' Will make interpolations of the wavefunctions.')
for i in range(kpproblem.size):
  if bnproblem[i] <= lastband:
    nk0 = kpproblem[i]
    nb0 = bnproblem[i]
    xx0 = np.full((7),-1,dtype=int)
    xx1 = np.full((7),-1,dtype=int)
    print(' ',nk0,'  ',nb0)
    print(neighbors[nk0,:])
    for j in range(4):
      nk = neighbors[nk0,j]
      if nk != -1 and nb0 <= lastband:
        if j == 0:
          xx0[2] = nk
        elif j == 1:
          xx1[2] = nk
        elif j == 2:
          xx0[4] = nk
        elif j == 3:
          xx1[4] = nk
        for jj in range(4):
          nk1 = neighbors[nk,jj]
          if nk1 != -1:
            if j == jj and j == 0:
              xx0[1] = nk1
              nk2 = neighbors[nk1,0]
              if nk2 != -1:
                xx0[0] = nk2
            elif j == jj and j == 1:
              xx1[1] = nk1
              nk2 = neighbors[nk1,1]
              if nk2 != -1:
                xx1[0] = nk2
            elif j == jj and j == 2:
              xx0[5] = nk1
              nk2 = neighbors[nk1,2]
              if nk2 != -1:
                xx0[6] = nk2
            elif j == jj and j == 3:
              xx1[5] = nk1
              nk2 = neighbors[nk1,3]
              if nk2 != -1:
                xx1[6] = nk2
 
 
    bx0 = np.full((7),-1,dtype=int)
    bx1 = np.full((7),-1,dtype=int)
 
    # Determine every pair k,b for the wfc used for the interpolation
    bx0 = bandsfinal[xx0,bnproblem[i]]
    bx1 = bandsfinal[xx1,bnproblem[i]]
 
    print('xx0',xx0)
    print('bx0',bx0)
    print('xx1',xx1)
    print('bx1',bx1)
 
 
    # nk0,nb0 - kpoint/machine band to be substituted
    print(' Interpolating ',nk0,nb0)
    interpolat.interpol(nr,nk0,nb0,xx0,xx1,bx0,bx1,wfcdirectory)


#sys.exit("Stop")
###################################################################################
print()
print(' *** Final Report ***')
print()
nrnotattrib = np.full((nbnd),-1,dtype=int)
sep = ' '
print(' Bands: gives the original band that belongs to new band (nk,nb)')
for nb in range(nbnd):
  nk = -1
  nrnotattrib[nb] = np.count_nonzero(bandsfinal[:,nb] == -1)
  print()
  print('  New band '+str(nb)+'      | y  x ->   nr of fails: '+str(nrnotattrib[nb]))
  for j in range(nky):
    lin = ''
    print()
    for i in range(nkx):
      nk = nk + 1
#      f = bands[nk,nb,sete]
      f = bandsfinal[nk,nb]
      if f < 0:
        lin += sep+sep+str(f)
      elif f>= 0 and f < 10:
        lin += sep+sep+sep+str(f)
      elif f > 9 and f < 100:
        lin += sep+sep+str(f)
      elif f > 99 and f < 1000:
        lin += sep+str(f)
    print(lin)
print()
print(' Signaling')
nrsignal = np.full((nbnd,7),-2,dtype=int)
for nb in range(nbnd):
  nk = -1
  nrsignal[nb,0] = str(np.count_nonzero(signalfinal[:,nb] == -1))
  nrsignal[nb,1]  = str(np.count_nonzero(signalfinal[:,nb] == 0))
  nrsignal[nb,2]  = str(np.count_nonzero(signalfinal[:,nb] == 1))
  nrsignal[nb,3]  = str(np.count_nonzero(signalfinal[:,nb] == 2))
  nrsignal[nb,4]  = str(np.count_nonzero(signalfinal[:,nb] == 3))
  nrsignal[nb,5]  = str(np.count_nonzero(signalfinal[:,nb] == 4))
  nrsignal[nb,6]  = str(np.count_nonzero(signalfinal[:,nb] == 5))
  print()
  print('     '+str(nb)+'      | y  x ->   -1: '+str(nrsignal[nb,0])+ '     0: '+str(nrsignal[nb,1]))
  for j in range(nky):
    lin = ''
    print()
    for i in range(nkx):
      nk = nk + 1
#      f = bands[nk,nb,sete]
      f = signalfinal[nk,nb]
      if f < 0:
        lin += sep+sep+str(f)
      elif f>= 0 and f < 10:
        lin += sep+sep+sep+str(f)
      elif f > 9 and nk < 100:
        lin += sep+sep+str(f)
      elif f > 99 and nk < 1000:
        lin += sep+str(f)
    print(lin)

print()
print(' Resume of results')
print()
print(' nr k-points not attributed to a band')
print(' Band       nr k-points')
for nb in range(nbnd):
  print(' ',nb,'         ',nrnotattrib[nb])

print()
print(' Signaling')
print(' Band  -1  0  1  2  3  4  5')
for nb in range(nbnd):
  print('  '+str(nb)+'   '+str(nrsignal[nb,:]))

print()

print(' Bands not usable (not completed)')
for nb in range(nbnd):
  if nrsignal[nb,1] != 0:
    print('  band ',nb,'  failed attribution of ',nrsignal[nb,1],' k-points')

print()
print(' Number of bands interpolated: ',kpproblem.shape)

print()


# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))

#sys.exit("Stop")

