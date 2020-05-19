# This is to include maths
import math
import re
import numpy as np
import ast
# This is to draw graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d
# This are the subroutines and functions
import load
# This is to make operations in the shell
import sys
import time
# This to make parallel processing
import joblib

if len(sys.argv)!=3:
  print(' ERROR in number of arguments. Has to have two integers.\n \
          If the first is negative, it will only calculate transitions between the too bands.')
  sys.exit("Stop")

bandfilled = int(sys.argv[1])                  # Number of the last filled band at k=0
bandempty = int(sys.argv[2])                   # Number of the last empty band at k=0

if bandfilled < 0:
  bandfilled = -bandfilled
  bandlist = [bandfilled,bandempty]
  print(' Calculating just transitions from band '+str(bandfilled)+' to '+str(bandempty))
else:
  bandlist = list(range(1,bandempty+1))
  print(' Calculating transitions from bands <'+str(bandfilled)+' to bands up to '+str(bandempty))

print(' List of bands: ',str(bandlist))

starttime = time.time()                        # Starts counting time

Ry = 13.6056923                                # Conversion factor from Ry to eV
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
eigenvalues = tmp[3]          # Dictionary with index k,band -> eigenvalue corrected for the new bands
                              # Energy in Ry
#sys.exit("Stop")

occ = load.readoccupancies(wfcdirectory,nbnd)    # {'58 6': 0.0, '25 8': 0.0, ...

apontador = load.readapontador(wfcdirectory,nbnd) # {'58 6': 8, '25 8': 8, '39 2': 3, ...

ban = load.readbandas(wfcdirectory,nbnd)             # kpoint,newband-> old band

berryConnection = {}

for i in range(1,bandempty+1):
  for j in range(1,bandempty+1):
    index = str(i)+' '+str(j)
    filename = wfcdirectory+'/berryCon'+str(i)+'-'+str(j)
    berryConnection[index] = joblib.load(filename+'.gz')   # Berry connection 

#print(berryConnection)
#sys.exit("Stop")
Earray = np.zeros((numero_kx,numero_ky,nbnd+1))
for j in range(numero_ky):
  for i in range(numero_kx):
    for banda in range(1,nbnd+1):
      index = str(kindex[str(i) + ' ' + str(j) + ' 0']) + ' ' + str(banda)
      if index in eigenvalues.keys():
        Earray[i,j,banda] = eigenvalues[index]

#      print(index,Earray[i,j,banda] )

print(' Finished reading data')
#sys.exit("Stop")

################################################## Finished reading data


enermax = 2.5            # Maximum energy (Ry)
enerstep = 0.001         # Energy step (Ry)
broadning = 0.01j        # energy broadning (Ry) 
const = 4*2j/(2*math.pi)**2            # = i2e^2/hslash 1/(2pi)^2     in Rydberg units
                                       # the '4' comes from spin degeneracy, that is summed in s and s'
vk = dk*dk/(2*math.pi)**2              # element of volume in k-space in units of bohr^-1

print(' Maximum energy (Ry): '+str(enermax))
print(' Energy step (Ry): '+str(enerstep))
print(' Energy broadning (Ry): '+str(np.imag(broadning)))
print(' Constant 4e^2/hslash 1/(2pi)^2     in Rydberg units: '+str(np.imag(const)))
print(' Volume (area) in k space: '+str(vk))

sigma = {}               # Dictionary where the conductivity will be stored
fermi = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1))
dE = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1))

for s in bandlist:
  for sprime in bandlist:
    dE[:,:,s,sprime] = Earray[:,:,s] - Earray[:,:,sprime]
    
    if s <= bandfilled  and sprime > bandfilled:
      fermi[:,:,s,sprime] = 1
    elif sprime <= bandfilled  and s > bandfilled:
      fermi[:,:,s,sprime] = -1

for omega in np.arange(0,enermax+enerstep,enerstep):
  omegaarray = np.full((numero_kx,numero_ky,bandempty+1,bandempty+1),omega+broadning)  # in Ry
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

