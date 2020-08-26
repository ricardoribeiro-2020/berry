###################################################################################
# This program reads the dot products and eigenvalues and establishes the bands
###################################################################################

# This is to include maths
import numpy as np
from random import randrange

# This is to make operations in the shell
import os
import sys
import time

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d


header('DOTPRODUCT',time.asctime())

starttime = time.time()                         # Starts counting time

if len(sys.argv) > 1:                           # To enter an initial value for k-point (optional)
  firskpoint = int(sys.argv[1])
else:
  firskpoint = -1

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
print(' Eigenvalues loaded')
#print(eigenvalues)

dp = np.loadtxt('dp.dat')
print(' Modulus of direct product loaded')

print(' Finished reading data')
##########################################################################
bands = np.full((nks,nbnd,100),-1,dtype=int)
for bnd in range(nbnd):
  bands[:,bnd,0] = bnd
#  print(bands[:,bnd,0])

#for nk in range(nks):
#  print(eigenvalues[nk,0])
connections = np.full((nks,4,nbnd,nbnd),-1.0)

#print(dp.shape[0])
for i in range(dp.shape[0]):
  nk1 = int(dp[i,0])
  nk2 = int(dp[i,1])
  b1  = int(dp[i,2]) - 1           # converts bands range from 1->nbdn to 0->nbdn-1
  b2 = int(dp[i,3]) - 1
  dotp = dp[i,4]
  for j in range(4):
    if nk2 == neighbors[nk1,j]:
      connections[nk1,j,b1,b2] = dotp

##########################################################################
tol = 0.9
ntentativ = 5                                   # Nr of tentatives
initialks = []                                  # List to save the initial k-points that are choosen randomly
signal = np.zeros((nks,nbnd,ntentativ+1),dtype=int)       # = nr of matches, -1 if contradictory matches
# Create arrays of tentatives         
for tentative in range(ntentativ):           
  if firskpoint >= 0 and firskpoint < nks and tentative == 0:
    kp0 = firskpoint
  else:
    kp0 = randrange(nks)                          # Chooses a k-point randomly
  initialks.append(kp0)                         # Stores the departure k-point for future reference
  listdone = [kp0]                              # List to store the k-points that have been analysed
  listk = []
  bands[kp0,:,tentative+1] = bands[kp0,:,0]     # initializes first k-point of the series
  signal[kp0,:,1] = 1                           # First k-point has signal 1
  for i in range(4):                            # Calculates the four points around the first
    for b1 in range(nbnd):
      for b2 in range(nbnd):
        if connections[kp0,i,b1,b2] > tol:      # Finds connections between k-points/bands
          bands[neighbors[kp0,i],b2,tentative+1] = bands[kp0,b1,tentative+1]
          signal[neighbors[kp0,i],b2,1] += 1      # Signal a connection
    if neighbors[kp0,i] not in listk and neighbors[kp0,i] not in listdone and neighbors[kp0,i] != -1:
      listk.append(neighbors[kp0,i])            # Adds neighbors not already done, for the next loop

  while len(listk) > 0:                         # Runs through the list of neighbors not already done
    nk = listk[0]                               # Chooses the first from the list

    for i in range(4):                          # Calculates the four points around
      for b1 in range(nbnd):
        for b2 in range(nbnd):
          if connections[nk,i,b1,b2] > tol:     # Finds connections between k-points/bands
            if bands[nk,b1,tentative+1] == -1 or \
               signal[neighbors[nk,i],b2,tentative+1] == -1:  # If that band is not valid, cycles the loop
              break
            if bands[neighbors[nk,i],b2,tentative+1] == -1:  # If the new band is not attributted, attribute it
              bands[neighbors[nk,i],b2,tentative+1] = bands[nk,b1,tentative+1]
              signal[neighbors[nk,i],b2,tentative+1] += 1      # Signal a connection
            elif bands[neighbors[nk,i],b2,tentative+1] == bands[nk,b1,tentative+1]:
              signal[neighbors[nk,i],b2,tentative+1] += 1      # Signal a connection
            else:
              signal[neighbors[nk,i],b2,tentative+1] = -1      # Signal a contradiction
 
      if neighbors[nk,i] not in listk and neighbors[nk,i] not in listdone and neighbors[nk,i] != -1:
        listk.append(neighbors[nk,i])
#      print(nk,i,bands[neighbors[nk,i],:,1])
    listk.remove(nk)                            # Remove k-point from the list of todo
    listdone.append(nk)                         # Add k-point to the list of done

#  print(tentative+1)
#  print(bands[:,:,tentative+1])
#  print(signal[:,:,tentative+1])


listdone.sort()
#print(listdone)
##########################################################################


bandsfinal = np.full((nks,nbnd),-1,dtype=int)        # Array for the final results
                                                     # gives the machine band that belongs to band (nk,nb)
signalfinal = np.zeros((nks,nbnd),dtype=int)         # Array for final signalling
first = True
attrib = []

cont = 1
for i in initialks:                        # Runs through all sets of bands
  if first:                                # First set is special
    first = False
    bandsfinal = bands[:,:,1]              # Starts with the first set
    signalfinal = signal[:,:,1].astype(int)            # Starts with first set signaling
    for nk in range(nks):
      if np.all(bandsfinal[nk,:] != -1) :
        attrib.append(nk)                # finds kpoints with all bands attributed in the first set
    continue
  cont += 1
#  print(bands[:,:,cont])
  for j in attrib:                         # Runs through all kpoints with all bands attributed
    if np.all(bands[j,:,cont] != -1):          # if the new set has also at the same kpoint all bands attributed, merge them
      if np.all(bands[j,:,1] == bands[j,:,cont]):
#        print(j,bands[j,:,cont],bands[j,:,1])

        print('***** Found same order! Merge.')
        for nk in range(nks):
          for nb in range(nbnd):
            if bandsfinal[nk,nb] == bands[nk,nb,cont] or bands[nk,nb,cont] == -1:
              continue
            elif bandsfinal[nk,nb] == -1 and bands[nk,nb,cont] != -1:
              bandsfinal[nk,nb] = bands[nk,nb,cont]
              signalfinal[nk,nb] = 1
              print(' Changed ',nk,nb)
            elif signalfinal[nk,nb] == -1:
              continue
            else:
              signalfinal[nk,nb] = -1
              print(nk,nb,bandsfinal[nk,nb],bands[nk,nb,cont])
              print(' !! Found incompatibility')

        break




#sys.exit("Stop")


print(' *** Final Report ***')
print()
nrnotattrib = np.full((nbnd),-1,dtype=int) 
sep = ' '
print(' Bands: gives the machine nr that belongs to new band (nk,nb)')
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
print(' Saving files apontador and bandas.')
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

print(' Saving files bandsfinal.npy and signalfinal.npy')
print(' bandsfinal.npy gives the machine number for each k-point/band')
with open('bandsfinal.npy', 'wb') as f:
  np.save(f,bandsfinal)
f.closed
with open('signalfinal.npy', 'wb') as f:
  np.save(f,signalfinal)
f.closed











# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))




