###################################################################################
# This program reads the wfc from DFT calculations make them coherent and saves
#  in separate files
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time

# This are the subroutines and functions
import contatempo
import dft
import loaddata as d

print()
print('     Program GENERATEWFC v.0.1 starts '+time.asctime())
print()
print('     This program is part of the open-source BERRY suite.')
print()

starttime = time.time()                         # Starts counting time

wfcdirectory = str(d.wfcdirectory)
dftdirectory = str(d.dftdirectory)

# Creates directory for wfc
os.system('mkdir -p '+wfcdirectory)
print(' Wavefunctions will be saved in directory',wfcdirectory)

nk = -1
nb = -1
if len(sys.argv)==1:
  print(' Will run for all k-points and bands')
  print(' There are',d.nks,'k-points and',d.nbnd,'bands.')
elif len(sys.argv)==2:
  nk = int(sys.argv[1])
  print(' Will run just for k-point',nk)
  print(' There are',d.nbnd,'bands.')
elif len(sys.argv)==3:
  nk = int(sys.argv[1])
  nb = int(sys.argv[2])
  print(' Will run just for k-point',nk,'and band',nb)

# Creates file with wfc of all bands at nk  ** DFT **
if nk == -1 and nb == -1:
  for nk in range(0,d.nks):
    for nb in range(1,d.nbnd+1):
      print(' Calculating wfc for k-point',nk,'and band',nb)
      dft.wfck2r(dftdirectory,wfcdirectory,nk,nb,d.npr)   

elif nk != -1 and nb == -1:
  for nb in range(1,d.nbnd+1):
    print(' Calculating wfc for k-point',nk,'and band',nb)
    dft.wfck2r(dftdirectory,wfcdirectory,nk,nb,d.npr)   

else:
  print(' Calculating wfc for k-point',nk,'and band',nb)
  dft.wfck2r(dftdirectory,wfcdirectory,nk,nb,d.npr)   










