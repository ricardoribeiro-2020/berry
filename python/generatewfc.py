###################################################################################
# This program reads the wfc from DFT calculations make them coherent and saves
#  in separate files
###################################################################################
# 
# Can accept 0, 1 or 2 arguments.
# If it has 0 arguments, it will run for all k-points and bands
# If it has 1 argument, it will run just for one k-point, specified by the argument
# If it has 2 arguments, it will run just for 1 k-point and 1 band, specified by the arguments

# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time

# This is for parallelism
import multiprocessing as mp

# This are the subroutines and functions
import contatempo
import dft
from headerfooter import header,footer
import loaddata as d

def extractconvert(nk):
  print(' Calculating wfc for k-point',nk)
  for nb in range(1,d.nbnd+1):
    dft.wfck2r(nk,nb)
  return

if __name__ == '__main__':
  header('GENERATEWFC',time.asctime())

  starttime = time.time()                         # Starts counting time

  wfcdirectory = str(d.wfcdirectory)
  dftdirectory = str(d.dftdirectory)

  # Creates directory for wfc
  os.system('mkdir -p '+wfcdirectory)
  print(' Wavefunctions will be saved in directory',wfcdirectory)
  print(' DFT files are in directory',dftdirectory)
  npr = d.npr
  print(' This program will run in '+str(npr)+' processors')
  print()
  nkx = d.nkx
  nky = d.nky
  nkz = d.nkz
  print(' Number of k-points in each direction:',nkx,nky,nkz)
  nks = d.nks
  print(' Total number of k-points:',nks)
  nr1 = d.nr1
  nr2 = d.nr2
  nr3 = d.nr3
  print(' Number of r-points in each direction:',nr1,nr2,nr3)
  nr = d.nr
  print(' Total number of points in real space:',nr)
  
  nbnd = d.nbnd
  print(' Number of bands:',nbnd)
  print()
  rpoint = int(nr1*nr2*1.1)
  print(' Point choosen for sincronizing phases: ',rpoint)
  print()
  ##########################################################################
  nk = -1
  nb = -1
  if len(sys.argv)==1:
    print(' Will run for all k-points and bands')
    print(' There are',nks,'k-points and',nbnd,'bands.')
  elif len(sys.argv)==2:
    nk = int(sys.argv[1])
    print(' Will run just for k-point',nk)
    print(' There are',nbnd,'bands.')
  elif len(sys.argv)==3:
    nk = int(sys.argv[1])
    nb = int(sys.argv[2])
    print(' Will run just for k-point',nk,'and band',nb)
  print()
#  npr=2
#  print(mp.cpu_count())
  # Creates file with wfc of all bands at nk  ** DFT **
  if nk == -1 and nb == -1:                  # Will run for all k-points and bands
    if npr > 1:
      with mp.Pool(processes=npr) as pool:
        pool.map(extractconvert, range(nks),int(npr))
    else:
      for nk in range(0,nks):
        print(' Calculating wfc for k-point',nk)
        for nb in range(1,d.nbnd+1):
          dft.wfck2r(nk,nb)
  
  elif nk != -1 and nb == -1:                # Will run just for k-point nk
    for nb in range(1,d.nbnd+1):
      print(' Calculating wfc for k-point',nk,'and band',nb)
      dft.wfck2r(nk,nb)
  
  else:                                      # Will run just for k-point nk and band nb
    print(' Calculating wfc for k-point',nk,'and band',nb)
    dft.wfck2r(nk,nb)
  
  



  # Finished
  endtime = time.time()
  
  footer(contatempo.tempo(starttime,endtime))
  


