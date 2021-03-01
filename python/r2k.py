###################################################################################
# This program reads a set of wavefunctions for different k and bands and translates that
# to another set for different points in real space, the functions become a function of k instead of r
###################################################################################

# This is to include maths
import numpy as np
from findiff import FinDiff, Gradient

# This is to make operations in the shell
import sys
import time

# These are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d

# This to save compressed files
import joblib

###################################################################################
if __name__ == '__main__':
  header('R2K',time.asctime())

  starttime = time.time()                         # Starts counting time

  if len(sys.argv)<2:
    print('     ERROR in number of arguments.')
    print('     Have to give the number of bands that will be considered.')
    print('     One number and calculates from 0 to that number.')
    print('     Two numbers and calculates from first to second.')
    sys.exit("Stop")
  elif len(sys.argv)==2:
    nbndmin = 0
    nbndmax = int(sys.argv[1]) + 1              # Number of bands to be considered
    print('     Will calculate bands and their gradient for bands 0 to',nbndmax)
  elif len(sys.argv)==3:
    nbndmin = int(sys.argv[1])                  # Number of lower band to be considered
    nbndmax = int(sys.argv[2]) + 1              # Number of higher band to be considered
    print('     Will calculate bands and their gradient for bands ',nbndmin,' to',nbndmax-1)
  
  # Reading data needed for the run

  wfcdirectory = str(d.wfcdirectory)
  print('     Directory where the wfc are:',wfcdirectory)
  nkx = d.nkx
  nky = d.nky
  nkz = d.nkz
  print('     Number of k-points in each direction:',nkx,nky,nkz)
  nks = d.nks
  print('     Total number of k-points:',nks)
  
  nr = d.nr
  print('     Total number of points in real space:',nr)
  npr = d.npr
  print('     Number of processors to use',npr)
  
  nbnd = d.nbnd
  print('     Number of bands:',nbnd)
  
  dk = float(d.step)            # Defines the step for gradient calculation dk
  print('     k-points step, dk',dk)
  print()

  kpoints = d.kpoints
  print('     kpoints loaded')      # kpoints = np.zeros((nks,3), dtype=float)
  
  r = d.r
  print('     rpoints loaded')      # r = np.zeros((nr,3), dtype=float)

  occupations = d.occupations
  print('     Occupations loaded')  # occupations = np.array(occupat)
  
  eigenvalues = d.eigenvalues
  print('     Eigenvalues loaded')  # eigenvalues = np.array(eigenval)

  phase = d.phase
  print('     Phases loaded')       # phase = np.zeros((nr,nks),dtype=complex)
  
  with open('ijltonk.npy', 'rb') as f:
    ijltonk = np.load(f)            # ijltonk converts kx,ky,kz to nk
  f.close()
  with open('bandsfinal.npy', 'rb') as f:
    bandsfinal = np.load(f)
  f.close()
  print('     bandsfinal.npy loaded')
  with open('signalfinal.npy', 'rb') as f:
    signalfinal = np.load(f)
  f.close()
  print('     signalfinal.npy loaded')
  print()
  sys.stdout.flush()
  #sys.exit("Stop")
  
################################################## Finished reading data

  grad = Gradient(h=[dk, dk],acc=2)            # Defines gradient function in 2D
################################################## 

  bandasr = {}                           # Dictionary with wfc for all points and bands
  bandasg = {}                           # Dictionary with wfc gradients for all points and bands
  
  for banda in range(nbndmin,nbndmax):   # For each band
    wfct_k = np.zeros((nr,nks),dtype=complex)
    for kp in range(nks):
      if signalfinal[kp,banda] == -1:           # if its a signaled wfc, choose interpolated
        infile= wfcdirectory+"/k0"+str(kp)+"b0"+str(bandsfinal[kp,banda])+".wfc1"
      else:                                     # else choose original
        infile = wfcdirectory+"/k0"+str(kp)+"b0"+str(bandsfinal[kp,banda])+".wfc"
      wfct_k[:,kp] = np.load(infile) 
    print('     Finished reading wfcs of band ',str(banda),'from files.   {:5.2f}'.format((time.time()-starttime)/60.),' min')
    sys.stdout.flush()

    wfcpos = {}                           # Dictionary with wfc for all points
    wfcgra = {}                           # Dictionary with wfc gradients for all points
  
    wfcpos = {posi: phase[posi,ijltonk[:,:,0]]*wfct_k[posi,ijltonk[:,:,0]] for posi in range(nr)}
    wfcgra = {posi: grad(wfcpos[posi]) for posi in range(nr)}
#    for posi in range(nr):
#      wfcpos[posi] = phase[posi,ijltonk[:,:,0]]*wfct_k[posi,ijltonk[:,:,0]]

#      wfcgra[posi] = grad(wfcpos[posi])                  # Complex gradient
  
    bandasr[banda] = wfcpos                              # add to dictionary
    bandasg[banda] = wfcgra
    print('     Finished band ',str(banda),'   {:5.2f}'.format((time.time()-starttime)/60.),' min')
    sys.stdout.flush()

    # Saving files
    joblib.dump(wfcpos,'./wfcpos'+str(banda)+'.gz', compress=3)
    joblib.dump(wfcgra,'./wfcgra'+str(banda)+'.gz', compress=3)
  

  sys.stdout.flush()

#sys.exit("Stop")


###################################################################################
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))



       



