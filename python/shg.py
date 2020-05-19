# This is to include maths
import math
import numpy as np
from findiff import FinDiff, coefficients, Coefficient, Gradient
#import ast
# This is to draw graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d
# This are the subroutines and functions
import load
from comutator import comute, comute3, deriv, comutederiv
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

directcomb,reverscomb = [],[]
berryConnection = {}                                 # Dictionary with the arrays of Berry connections
for i in range(1,bandempty+1):
  for j in range(1,bandempty+1):
    index = str(i)+' '+str(j)
    filename = wfcdirectory+'/berryCon'+str(i)+'-'+str(j)
    berryConnection[index] = joblib.load(filename+'.gz')   # Berry connection filled s, empty s'

#print(berryConnection.keys())
#print(berryConnection)
#sys.exit("Stop")
Earray = np.zeros((numero_kx,numero_ky,bandempty+1))
for j in range(numero_ky):
  for i in range(numero_kx):
    for banda in range(1,bandempty+1):
      index = str(kindex[str(i) + ' ' + str(j) + ' 0']) + ' ' + str(banda)
      if index in eigenvalues.keys():
        Earray[i,j,banda] = eigenvalues[index]

#      print(index,Earray[i,j,banda] )

print(' Finished reading data')
#sys.exit("Stop")

################################################## Finished reading data

grad = Gradient(h=[dk, dk],acc=3)            # Defines gradient function in 2D
################################################## 


enermax = 2.5            # Maximum energy (Ry)
enerstep = 0.001         # Energy step (Ry)
broadning = 0.01j        # energy broadning (Ry) 
const = 2*math.sqrt(2)*2/(2*math.pi)**2      # = -2e^3/hslash 1/(2pi)^2     in Rydberg units
                                             # the 2e comes from having two electrons per band
                                             # another minus comes from the negative charge
vk = dk*dk/(2*math.pi)**2              # element of volume in k-space in units of bohr^-1
                                       # it is actually an area, because we have a 2D crystal

sigma = {}               # Dictionary where the conductivity will be stored
fermi = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1))
dE = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1))
graddE = np.zeros((2,numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex) 
gamma1 = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex)
gamma2 = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex)
gamma3 = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex)
gamma12 = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex)
gamma13 = np.zeros((numero_kx,numero_ky,bandempty+1,bandempty+1),dtype=complex)

for s in bandlist:
  for sprime in bandlist:
    dE[:,:,s,sprime] = Earray[:,:,s] - Earray[:,:,sprime]
    graddE[:,:,:,s,sprime] = grad(dE[:,:,s,sprime])
    if s <= bandfilled  and sprime > bandfilled:
      fermi[:,:,s,sprime] = 1
    elif sprime <= bandfilled  and s > bandfilled:
      fermi[:,:,s,sprime] = -1

#    print(dE[:,:,s,sprime])
#    print(graddE[:,:,:,s,sprime])
#e = comute(berryConnection,s,sprime,alpha,beta)

for omega in np.arange(0,enermax+enerstep,enerstep):
  omegaarray = np.full((numero_kx,numero_ky,bandempty+1,bandempty+1),omega+broadning)  # in Ry
  sig = np.full((numero_kx,numero_ky,2,2,2),0.+0j,dtype=complex) # matrix sig_xxx,sig_xxy,...,sig_yyx,sig_yyy

  gamma1 = const*dE/(2*omegaarray-dE)                      # factor called dE/g in paper times leading constant
  gamma2 = -fermi/np.square(omegaarray-dE)                  # factor f/h^2 in paper (-) to account for change in indices in f and h
  gamma3 = -fermi/(omegaarray-dE)                           # factor f/h in paper (index reference is of h, not f, in equation)

  for s in bandlist:                       # runs through index s
    for sprime in bandlist:                # runs through index s'
      gamma12[:,:,s,sprime] = gamma1[:,:,s,sprime]*gamma2[:,:,s,sprime]
      gamma13[:,:,s,sprime] = gamma1[:,:,s,sprime]*gamma3[:,:,s,sprime]

#  sys.exit("Stop")

  for beta in range(2):                  # beta is spatial coordinate
    for alpha1 in range(2):               # alpha1 is spatial coordinate
      for alpha2 in range(2):               # alpha2 is spatial coordinate

        for s in bandlist:                       # runs through index s
          for sprime in bandlist:                # runs through index s'
            if s == sprime:
              continue
            sig[:,:,beta,alpha1,alpha2] += \
               (graddE[alpha2,:,:,s,sprime]*comute(berryConnection,sprime,s,beta,alpha1) \
              + graddE[alpha1,:,:,s,sprime]*comute(berryConnection,sprime,s,beta,alpha2) \
               )*gamma12[:,:,s,sprime]*0.5 

            sig[:,:,beta,alpha1,alpha2] += \
               (comutederiv(berryConnection,s,sprime,beta,alpha1,alpha2,dk) \
#              + comutederiv(berryConnection,s,sprime,beta,alpha2,alpha1,dk) \  # to include, divide by 2.
               )*gamma13[:,:,s,sprime]      
      
      
            for r in bandlist:                   # runs through index r
              if r == sprime or r == s:
                continue
              sig[:,:,beta,alpha1,alpha2] += -0.25j*gamma1[:,:,s,sprime]*   \
                 (comute3(berryConnection,sprime,s,r,beta,alpha2,alpha1)    \
                + comute3(berryConnection,sprime,s,r,beta,alpha1,alpha2))*gamma3[:,:,r,sprime] \
                -(comute3(berryConnection,sprime,s,r,beta,alpha1,alpha2)    \
                + comute3(berryConnection,sprime,s,r,beta,alpha2,alpha1))*gamma3[:,:,s,r] 

#    print(sig)

  sigma[omega] = np.sum(np.sum(sig,axis=0),axis=0)*vk

#  sys.exit("Stop")

with open('sigma2r.dat','w') as sigm:
  sigm.write('# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, \
                             sigma_yyx, sigma_yxy, sigma_yxx\n')
  for omega in np.arange(0,enermax+enerstep,enerstep):
    outp = '{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n'
    sigm.write(outp.format(omega*Ry,np.real(sigma[omega][0,0,0]),np.real(sigma[omega][1,1,1]),\
                                    np.real(sigma[omega][0,0,1]),np.real(sigma[omega][0,1,0]),\
                                    np.real(sigma[omega][0,1,1]),np.real(sigma[omega][1,1,0]),\
                                    np.real(sigma[omega][1,0,1]),np.real(sigma[omega][1,0,0])))
sigm.closed

with open('sigma2i.dat','w') as sigm:
  sigm.write('# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, \
                             sigma_yyx, sigma_yxy, sigma_yxx\n')
  for omega in np.arange(0,enermax+enerstep,enerstep):
    outp = '{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n'
    sigm.write(outp.format(omega*Ry,np.imag(sigma[omega][0,0,0]),np.imag(sigma[omega][1,1,1]),\
                                    np.imag(sigma[omega][0,0,1]),np.imag(sigma[omega][0,1,0]),\
                                    np.imag(sigma[omega][0,1,1]),np.imag(sigma[omega][1,1,0]),\
                                    np.imag(sigma[omega][1,0,1]),np.imag(sigma[omega][1,0,0])))
sigm.closed


#sys.exit("Stop")

