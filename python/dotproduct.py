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


def connection(nk,neighbor,dphase):
# Calculates the dot product  of all combinations of wfc in nk and neighbor
  wfcdirectory = str(d.wfcdirectory)
  tol0 = 0.9
  tol1 = 0.85
  nbnd = int(d.nbnd)
  dpc1 = np.zeros((nbnd,nbnd),dtype=complex)
  dpc2 = np.zeros((nbnd,nbnd),dtype=complex)
  for banda0 in range(nbnd):

    infile = wfcdirectory+'k0'+str(nk)+'b0'+str(banda0)+'.wfc'
#    print(infile)
    with open(infile, 'rb') as f:
      wfc0 = np.load(f)
    f.close()
#    print(wfc0)
    for banda1 in range(nbnd):

      infile = wfcdirectory+'k0'+str(neighbor)+'b0'+str(banda1)+'.wfc'

#      print(infile)

      with open(infile, 'rb') as f:
        wfc1 = np.load(f)
      f.close()
#      print(wfc1)

      dpc1[banda0,banda1] = np.sum(dphase*wfc0*np.conjugate(wfc1))
      dpc2[banda1,banda0] = np.conjugate(dpc1[banda0,banda1] )

#  print(dpc1)

  return dpc1,dpc2

##########################################################
if __name__ == '__main__':
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
  ##########################################################
  
  dpc = np.full((nks,4,nbnd,nbnd),0+0j,dtype=complex)
  dp = np.zeros((nks,4,nbnd,nbnd))
  
  for nk in range(nks):                    # runs through all k-points
    for j in range(4):                     # runs through all neighbors
      neighbor = neighbors[nk,j]

      if neighbor != -1 and neighbor > nk:            # exclude invalid neighbors
  #      print(nk,j,neighbors[nk,j])
        jNeighbor = np.where(neighbors[neighbor]==nk)

        dphase = phase[:,nk]*np.conjugate(phase[:,neighbor])
#        print(dphase)
        print("Calculating   nk = "+str(nk)+"  neighbor = "+str(neighbor))
        sys.stdout.flush()
        #dpc[nk,j,:,:] = connection(nk,neighbor,dphase)
        dpc[nk,j,:,:],dpc[neighbor,jNeighbor,:,:] = connection(nk,neighbor,dphase)
#        print(dpc[nk,j,:,:])
#        sys.exit("Stop")
  
  dp = np.abs(dpc)

  # Save dot products to file
  with open('dpc.npy', 'wb') as f:
    np.save(f,dpc)
  f.close()
  print(' Dot products saved to file dpc.npy')

  # Save dot products modulus to file
  with open('dp.npy', 'wb') as f:
    np.save(f,dp)
  f.close()
  print(' Dot products modulus saved to file dp.npy')



  # Finished
  endtime = time.time()
  
  footer(contatempo.tempo(starttime,endtime))




