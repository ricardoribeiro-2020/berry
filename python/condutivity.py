###################################################################################
# This program calculates the linear conductivity from the Berry connections
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import sys
import time

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d

# This to make parallel processing
import joblib

###################################################################################
if __name__ == '__main__':
  header('CONDUTIVITY',time.asctime())

  starttime = time.time()                         # Starts counting time


  if len(sys.argv)<3:
    print(' ERROR in number of arguments. Has to have two integers.\n \
            If the first is negative, it will only calculate transitions between the too bands.')
    sys.exit("Stop")
  elif len(sys.argv)==3:
    bandfilled = int(sys.argv[1])                  # Number of the last filled band at k=0
    bandempty = int(sys.argv[2])                   # Number of the last empty band at k=0
    inputfile = ''
  elif len(sys.argv)==4:
    bandfilled = int(sys.argv[1])                  # Number of the last filled band at k=0
    bandempty = int(sys.argv[2])                   # Number of the last empty band at k=0
    inputfile = str(sys.argv[3])                   # Name of the file where data for the graphic is

  if bandfilled < 0:
    bandfilled = -bandfilled
    bandlist = [bandfilled,bandempty]
    print(' Calculating just transitions from band '+str(bandfilled)+' to '+str(bandempty))
  else:
    bandlist = list(range(bandempty+1))
    print(' Calculating transitions from bands <'+str(bandfilled)+' to bands up to '+str(bandempty))
  
  print(' List of bands: ',str(bandlist))

# Default values:
  enermax = 2.5            # Maximum energy (Ry)
  enerstep = 0.001         # Energy step (Ry)
  broadning = 0.01j        # energy broadning (Ry) 
  if inputfile != '':
    with open(inputfile,'r') as le:
      inputvar = le.read().split("\n")
    le.close()
    # Read that from input file
    for i in inputvar:
      ii = i.split()
      if len(ii) == 0:
        continue
      if ii[0] == 'enermax':
        enermax = float(ii[1])
      if ii[0] == 'enerstep':
        enerstep = float(ii[1])
      if ii[0] == 'broadning':
        broadning = 1j*float(ii[1])



  Ry = 13.6056923                                # Conversion factor from Ry to eV
################################################ Read data
  print(' Start reading data')

# Reading data needed for the run

  nkx = d.nkx
  nky = d.nky
  nkz = d.nkz
  print(' Number of k-points in each direction:',nkx,nky,nkz)
  nbnd = d.nbnd
  print(' Number of bands:',nbnd)
  dk = float(d.step)            # Defines the step for gradient calculation dk
  print(' k-points step, dk',dk)
  print()
  occupations = d.occupations
  print(' occupations loaded')  # occupations = np.array(nks,nbnd)
  eigenvalues = d.eigenvalues
  print(' eigenvalues loaded')  # eigenvalues = np.array(nks,nbnd)
  with open('bandsfinal.npy', 'rb') as f:
    bandsfinal = np.load(f)
  f.closed
  print(' bandsfinal.npy loaded')
  with open('signalfinal.npy', 'rb') as f:
    signalfinal = np.load(f)
  f.closed
  print(' signalfinal.npy loaded')
  print()
  
  sys.stdout.flush()
  #sys.exit("Stop")
  
  berryConnection = {}
  for i in range(bandempty+1):
    for j in range(bandempty+1):
      index = str(i)+' '+str(j)
      filename = './berryCon'+str(i)+'-'+str(j)
      berryConnection[index] = joblib.load(filename+'.gz')   # Berry connection 
  
  #sys.exit("Stop")
  Earray = np.zeros((nkx,nky,nbnd))     # Eigenvalues corrected for the new bands
  
  kp = 0
  for j in range(nky):                  # Energy in Ry
    for i in range(nkx):
      for banda in range(nbnd):
        Earray[i,j,banda] = eigenvalues[kp,bandsfinal[kp,banda]]
      kp += 1
  #      print(Earray[i,j,banda] )
  
  print(' Finished reading data')
  #sys.exit("Stop")
  
################################################## Finished reading data
  
  
  const = 4*2j/(2*np.pi)**2            # = i2e^2/hslash 1/(2pi)^2     in Rydberg units
                                         # the '4' comes from spin degeneracy, that is summed in s and s'
  vk = dk*dk/(2*np.pi)**2              # element of volume in k-space in units of bohr^-1
  
  print(' Maximum energy (Ry): '+str(enermax))
  print(' Energy step (Ry): '+str(enerstep))
  print(' Energy broadning (Ry): '+str(np.imag(broadning)))
  print(' Constant 4e^2/hslash 1/(2pi)^2     in Rydberg units: '+str(np.imag(const)))
  print(' Volume (area) in k space: '+str(vk))
  
  sigma = {}               # Dictionary where the conductivity will be stored
  fermi = np.zeros((nkx,nky,bandempty+1,bandempty+1))
  dE = np.zeros((nkx,nky,bandempty+1,bandempty+1))
  
  
  for s in bandlist:
    for sprime in bandlist:
      dE[:,:,s,sprime] = Earray[:,:,s] - Earray[:,:,sprime]
      
      if s <= bandfilled  and sprime > bandfilled:
        fermi[:,:,s,sprime] = 1
      elif sprime <= bandfilled  and s > bandfilled:
        fermi[:,:,s,sprime] = -1


#sys.exit("Stop")
  for omega in np.arange(0,enermax+enerstep,enerstep):
    omegaarray = np.full((nkx,nky,bandempty+1,bandempty+1),omega+broadning)  # in Ry
    sig = np.full((2,2),0.+0j)                                # matrix sig_xx,sig_xy,sig_yy,sig_yx
  
    gamma = const*dE/(omegaarray-dE)                           # factor that multiplies
  
    for s in bandlist:                       # runs through index s
      for sprime in bandlist:                # runs through index s'
        if s == sprime:
          continue
        s_sprime = str(s) + ' ' + str(sprime)
        sprime_s = str(sprime) + ' ' + str(s)
 
        for beta in range(2):                  # beta is spatial coordinate
          for alpha in range(2):               # alpha is spatial coordinate

            sig[alpha,beta] += np.sum(gamma[:,:,sprime,s]* berryConnection[s_sprime][alpha]*berryConnection[sprime_s][beta]*fermi[:,:,s,sprime])

    sigma[omega] = sig*vk


  with open('sigmar.dat','w') as sigm:
    sigm.write('# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n')
    for omega in np.arange(0,enermax+enerstep,enerstep):
      outp = '{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}\n'
      sigm.write(outp.format(omega*Ry,np.real(sigma[omega][0,0]),np.real(sigma[omega][1,1]),\
                                     np.real(sigma[omega][1,0]),np.real(sigma[omega][0,1])))
  sigm.closed

  with open('sigmai.dat','w') as sigm:
    sigm.write('# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n')
    for omega in np.arange(0,enermax+enerstep,enerstep):
      outp = '{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}\n'
      sigm.write(outp.format(omega*Ry,np.imag(sigma[omega][0,0]),np.imag(sigma[omega][1,1]),\
                                      np.imag(sigma[omega][1,0]),np.imag(sigma[omega][0,1])))
  sigm.closed


#sys.exit("Stop")

###################################################################################
# Finished
  endtime = time.time()

  footer(contatempo.tempo(starttime,endtime))




