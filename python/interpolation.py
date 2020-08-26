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

##########################################################################

# Select points signaled -1
problem = np.where(signalfinal == -1)
kpproblem = problem[0]
bnproblem = problem[1]

# Create array with (kp,b1,b2) with the two problematic bands of each k-point
# in array kpproblem, the k-points are in pairs, so make use of it
problemlength = int(kpproblem.size/2)
kpb1b2 = np.zeros((problemlength,3),dtype=int)

print()
print(' Report on problems found')
print()
for i in range(problemlength):
  kpb1b2[i,0] = kpproblem[i*2]
  kpb1b2[i,1] = bnproblem[i*2]
  kpb1b2[i,2] = bnproblem[i*2+1]

  print(' Neighbors of the k-point with problem: ',kpb1b2[i,0],neighbors[kpb1b2[i,0],:])
  print(bandsfinal[kpb1b2[i,0],:])
  validneig = np.count_nonzero(neighbors[kpb1b2[i,0],:] != -1)
  count11 = count12 = count21 = count22 = 0
  for neig in range(4):
    if neighbors[kpb1b2[i,0],neig] != -1:
      print(bandsfinal[neighbors[kpb1b2[i,0],neig],:])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2],bnproblem[i*2], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2]])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2],bnproblem[i*2+1], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2+1]])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2+1],bnproblem[i*2], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2]]) 
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2+1],bnproblem[i*2+1], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2+1]])

      if connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2]] > 0.85:
        count11 += 1
      if connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2+1]] > 0.85:
        count12 += 1
      if connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2]] > 0.85:
        count21 += 1
      if connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2+1]] > 0.85:
        count22 += 1


  if count11 == validneig:
    signalfinal[kpb1b2[i,0],bnproblem[i*2]] = validneig                  # signals problem as solved
    signalfinal[kpb1b2[i,0],bnproblem[i*2+1]] = validneig                # signals problem as solved

  if count12 == validneig:
    signalfinal[kpb1b2[i,0],bnproblem[i*2]] = validneig                  # signals problem as solved
    signalfinal[kpb1b2[i,0],bnproblem[i*2+1]] = validneig                # signals problem as solved

  if count21 == validneig:
    signalfinal[kpb1b2[i,0],bnproblem[i*2]] = validneig                  # signals problem as solved
    signalfinal[kpb1b2[i,0],bnproblem[i*2+1]] = validneig                # signals problem as solved

  if count22 == validneig:
    signalfinal[kpb1b2[i,0],bnproblem[i*2]] = validneig                  # signals problem as solved
    signalfinal[kpb1b2[i,0],bnproblem[i*2+1]] = validneig                # signals problem as solved

print()
print(' Cases where attribution failed')
print()
# Select points signaled 0
kpproblem,bnproblem = np.where(signalfinal == 0)
problemlength = int(kpproblem.size)
kpb1b2 = np.zeros((problemlength,2),dtype=int)
for i in range(problemlength):
  kpb1b2[i,0] = kpproblem[i]
  kpb1b2[i,1] = bnproblem[i]
  validneig = np.count_nonzero(neighbors[kpb1b2[i,0],:] != -1)
  count11 = 0
  refbnd = -1
  for neig in range(4):
    if neighbors[kpb1b2[i,0],neig] != -1:
      for j in range(nbnd):
        if connections[kpb1b2[i,0],neig,bnproblem[i],bandsfinal[neighbors[kpb1b2[i,0],neig],j]] > 0.8 and bandsfinal[neighbors[kpb1b2[i,0],neig],j] !=-1:
          print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i],bandsfinal[neighbors[kpb1b2[i,0],neig],j], \
               connections[kpb1b2[i,0],neig,bnproblem[i],bandsfinal[neighbors[kpb1b2[i,0],neig],j]])
          if refbnd == -1:
            refbnd = bandsfinal[neighbors[kpb1b2[i,0],neig],j]
            count11 +=1
          elif refbnd == bandsfinal[neighbors[kpb1b2[i,0],neig],j]:
            count11 +=1
          else:
            count11 = -100
  if count11 > 0:
    print(' Found!')
    bandsfinal[kpb1b2[i,0],kpb1b2[i,1]] = refbnd
    signalfinal[kpb1b2[i,0],kpb1b2[i,1]] = count11




print()
print(' Problems not solved')
problem = np.where(signalfinal == -1)
kpproblem = problem[0]
bnproblem = problem[1]

problemlength = int(kpproblem.size/2)
kpb1b2 = np.zeros((problemlength,3),dtype=int)
for i in range(problemlength):
  kpb1b2[i,0] = kpproblem[i*2]
  kpb1b2[i,1] = bnproblem[i*2]
  kpb1b2[i,2] = bnproblem[i*2+1]

  print(' Neighbors of the k-point with problem: ',neighbors[kpb1b2[i,0],:])
  print(bandsfinal[kpb1b2[i,0],:])
  validneig = np.count_nonzero(neighbors[kpb1b2[i,0],:] != -1)
  count11 = 0
  count12 = 0
  count21 = 0
  count22 = 0
  for neig in range(4):
    if neighbors[kpb1b2[i,0],neig] != -1:
      print(bandsfinal[neighbors[kpb1b2[i,0],neig],:])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2],bnproblem[i*2], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2]])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2],bnproblem[i*2+1], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2],bnproblem[i*2+1]])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2+1],bnproblem[i*2], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2]])
      print(kpb1b2[i,0],neighbors[kpb1b2[i,0],neig],bnproblem[i*2+1],bnproblem[i*2+1], \
               connections[kpb1b2[i,0],neig,bnproblem[i*2+1],bnproblem[i*2+1]])






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
      elif f > 9 and nk < 100:
        lin += sep+sep+str(f)
      elif f > 99 and nk < 1000:
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
print(' Saving files new apontador and bandas.')
with open(wfcdirectory+'/apontador','w') as aa:      # Reads pointers of bands from file
  for nb in range(nbnd):
    for nk in range(nks):
      aa.write('  ' + str(nk) + '  ' + str(bandsfinal[nk,nb]) + '  ' + str(nb) + '  ' + str(signalfinal[nk,nb]) + '\n')
aa.closed
with open(wfcdirectory+'/bandas','w') as ba:
  for nb in range(nbnd):
    for nk in range(nks):
      ba.write('  ' + str(nb) + '  ' + str(nk) + '  ' + str(bandsfinal[nk,nb]) + '\n')
ba.closed

print(' Saving new files bandsfinal.npy and signalfinal.npy')
print(' bandsfinal.npy gives the machine number for each k-point/band')
with open('bandsfinal1.npy', 'wb') as f:
  np.save(f,bandsfinal)
f.closed
with open('signalfinal1.npy', 'wb') as f:
  np.save(f,signalfinal)
f.closed




# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))

#sys.exit("Stop")

