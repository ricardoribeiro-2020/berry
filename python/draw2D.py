# This program draws the bands
# This is to include maths
import math
import numpy as np
# This is to make operations in the shell
import sys
# This is to draw graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d
# This are the subroutines and functions
import load
# This to make parallel processing
import joblib

if len(sys.argv)!=3:
  print(' ERROR in number of arguments. Has to have two integers.')
  print(' The first is the first band, the second is last band.')
  sys.exit("Stop")

startband = int(sys.argv[1])                  # Number of the first band
endband = int(sys.argv[2])                    # Number of the last band

fig = plt.figure(figsize=(6,6))
wfcdirectory = 'wfc'                            # Directory to store the final wfc
cores = ['black','blue','green','red','grey','brown','violet',\
         'seagreen','dimgray','darkorange','royalblue','darkviolet','maroon',\
         'yellowgreen','peru','steelblue','crimson','silver','magenta','yellow']

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

apontador = load.readapontador(wfcdirectory,nbnd) # {'58 6': 8, '25 8': 8, '39 2': 3, ...

xarray = np.zeros((numero_kx,numero_ky))
yarray = np.zeros((numero_kx,numero_ky))
zarray = np.zeros((numero_kx,numero_ky))
count = -1
for j in range(numero_ky):
  for i in range(numero_kx):
    count = count + 1
    xarray[i,j] = kpointx[count]
    yarray[i,j] = kpointy[count]

ax = fig.gca(projection='3d')
for banda in range(startband,endband+1):
  count = -1
  for j in range(numero_ky):
    for i in range(numero_kx):
      count = count + 1
      index = str(count) + ' ' + str(banda)
      if index in eigenvalues.keys():
        zarray[i,j] = eigenvalues[index]

  ax.plot_wireframe(xarray, yarray, zarray, color=cores[banda-1])

# Para desenhar no mathematica!
#
# print('b'+str(banda)+'={', end = '')
# for beta in range(numero_ky):
#   print('{', end = '')
#   for alfa in range(numero_kx):
#     if alfa != numero_kx-1:
#       print(str(zarray[alfa][beta])+str(','), end = '')
#     else:
#       print(str(zarray[alfa][beta]), end = '')
#   if beta != numero_ky-1:
#     print('},')
#   else:
#     print('}', end = '')
# print('};\n')


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

  #ax.plot_trisurf(xarray, yarray, zarray, linewidth=0.2, antialiased=True)

plt.show()

