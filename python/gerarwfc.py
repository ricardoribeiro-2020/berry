# This program runs a set of dft calculations, extracts the resulting wavefunctions
# and order them by bands

# This is to include maths
import math
import re
import numpy as np
# This is to make operations in the shell
import sys
import os
import time
# This are the subroutines and functions
import dft
import readwfck2r
import contatempo

print('******* Program gerarwfc.py ********')
print('*                                  *')
print()

starttime = time.time()                         # Starts counting time
# Defines some constants
np = 12
#mpi = ''
mpi = 'mpirun -np ' + str(np) + ' '

directory = './dft/'                            # Files names and directory
name_scf = 'scf'
name_nscf = 'nscf'
wfcdirectory = 'wfc'                            # Directory to store the final wfc
os.popen('mkdir -p '+ wfcdirectory)             # Creates the directory if not exist

print('* Important variables of the run *')
nbands = 30
k0 = [0.00, 0.00, 0.00]                         # Defines departure k-point
numero_kx = 47                                  # Number of k-points in each direction
numero_ky = 41
numero_kz = 1

step = 0.005                                     # Step between k-points
create = True                                   # True if creates wfc files, false if already created
limite = 220                                     # half the number of points in z direction (around z=0)

print(' Number of bands in the nscf calculation ',str(nbands))
print(' Starting k-point of the mesh ',str(k0))
print(' Number of points in the mesh ',str(numero_kx),str(numero_ky),str(numero_kz))
print(' Step of the k-point mesh ',str(step))
print(' Number of points in z direction for wfc in real space ',str(2*limite))
print()
sys.stdout.flush()

dft.scf(mpi,directory,name_scf)                 # Runs scf calculation  ** DFT **
nscf = dft.template(directory,name_scf)         # Creates template for nscf calculation  ** DFT **

kpoints = ''                                    # to append to nscf file
k = []                                          # k-points list 
kindex = {}                                     # index of k-points: first is 0
kIntindex = {}                                  # index of k-points with integers
nkps = 0                                        # count the k-points
for l in range(numero_kz):
  for j in range(numero_ky):
    for i in range(numero_kx):
      nkps = nkps +1
      index = (str(i)+' '+str(j)+' '+str(l))
      k1, k2, k3 = round(k0[0] + step*i,6), round(k0[1] + step*j,6), round(k0[2] + step*l,6)
      kkk = "{:.5f}".format(k1) + '  ' + "{:.5f}".format(k2) + '  ' + "{:.5f}".format(k3)
      kpoints = kpoints + kkk + '  1\n'
      k.append([kkk])
      kindex[kkk] = nkps - 1
      kIntindex[index] = nkps - 1

dft.nscf(mpi,directory,name_nscf,nscf,nkps,kpoints,nbands)  # Runs nscf calculations for all k-points  ** DFT **
sys.stdout.flush()

##################################################
# wfc @ nk = 0
##################################################
dft.wfck2r(directory,0,wfcdirectory,create,limite)  # Creates file with wfc of all bands at nk=0  ** DFT **
nbnd = readwfck2r.numberBands()                     # Returns the number of bands
eig = readwfck2r.eigenvalues()                      # Returns a list with the eigenvalues for this k-point
eig.insert(0,0)                                     # Inserts reference to k-point
enerBands = [eig]                                   # Creates list of eigenvalues
occup = readwfck2r.occupancies()                    # Returns a list with the occupancies for this k-point
occup.insert(0,0)                                   # Inserts reference to k-point
occ = [occup]                                       # Creates list of occupancies

app = {}                                            # Creates dictionary for pointers
for i in range(1,nbnd+1):                
  index = str(0)+' '+str(i)
  app[index] = i
                                        
print(' Finished creating wfc 0   '+str((time.time()-starttime)/60.),' min')
print()
sys.stdout.flush()

dege = list(range(1, nbnd+1))                       # defines list of bands to compare: original order
alert = False

##################################################
# wfc @ nk = 1...nkps
##################################################
for nk in range(1,nkps):                                # Loop through all other k-points
  dft.wfck2r(directory,nk,wfcdirectory,create,limite)   # Creates file with wfc of all bands at nk  ** DFT **
  sys.stdout.flush()
  eig = readwfck2r.eigenvalues()                        # Returns a list with the eigenvalues for this k-point
  eig.insert(0,nk)
  enerBands.append(eig)
  occup = readwfck2r.occupancies()                      # Returns a list with the occupancies for this k-point
  occup.insert(0,nk)
  occ.append(occup)


##################################################
with open(wfcdirectory+'/k_points','w') as ks:
  for item in k:
    for it in item:
      ks.write(str(it)+'   ')
    ks.write('\n')
ks.closed
with open(wfcdirectory+'/eigenvalues','w') as ei:
  for item in enerBands:
    for it in item:
      ei.write(str(it)+'   ')
    ei.write('\n')
ei.closed

with open(wfcdirectory+'/kindex','w') as ki:
  ki.write(str(numero_kx)+' '+str(numero_ky)+' '+str(numero_kz)+' \n')
  for l in range(numero_kz):
    for j in range(numero_ky):
      for i in range(numero_kx):
        index = (str(i)+' '+str(j)+' '+str(l))
        ki.write(str(index)+'  '+str(kIntindex[index])+'\n')
ki.closed

with open(wfcdirectory+'/occupancies','w') as oc:
  for item in occ:
    for it in item:
      oc.write(str(it)+'   ')
    oc.write('\n')
oc.closed


endtime = time.time()
print(contatempo.tempo(starttime,endtime))



