###################################################################################
# This program draws the bands
###################################################################################

# This is to include maths
import numpy as np

# This is to make operations in the shell
import sys
import time

# This is to draw graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d

# This are the subroutines and functions
import contatempo
from headerfooter import header,footer
import loaddata as d

header('DRAWBANDS',time.asctime())

starttime = time.time()                         # Starts counting time

if len(sys.argv)!=3:
  print(' ERROR in number of arguments. Has to have two integers.')
  print(' The first is the first band, the second is last band.')
  sys.exit("Stop")

startband = int(sys.argv[1])                  # Number of the first band
endband = int(sys.argv[2])                    # Number of the last band

fig = plt.figure(figsize=(6,6))

cores = ['black','blue','green','red','grey','brown','violet',\
         'seagreen','dimgray','darkorange','royalblue','darkviolet','maroon',\
         'yellowgreen','peru','steelblue','crimson','silver','magenta','yellow']

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
nbnd = d.nbnd
print(' Number of bands:',nbnd)
print()
eigenvalues = d.eigenvalues
print(' Eigenvlaues loaded')
kpoints = d.kpoints
print(' K-points loaded')

with open('bandsfinal1.npy', 'rb') as f:
  bandsfinal = np.load(f)
f.closed
print(' bandsfinal loaded')

xarray = np.zeros((nkx,nky))
yarray = np.zeros((nkx,nky))
zarray = np.zeros((nkx,nky))
count = -1
for j in range(nky):
  for i in range(nkx):
    count = count + 1
    xarray[i,j] = kpoints[count,0]
    yarray[i,j] = kpoints[count,1]

ax = fig.gca(projection='3d')
for banda in range(startband,endband+1):
  count = -1
  for j in range(nky):
    for i in range(nkx):
      count = count + 1
      zarray[i,j] = eigenvalues[count,bandsfinal[count,banda]]

  ax.plot_wireframe(xarray, yarray, zarray, color=cores[banda])

# Para desenhar no mathematica!
#
# print('b'+str(banda)+'={', end = '')
# for beta in range(nky):
#   print('{', end = '')
#   for alfa in range(nkx):
#     if alfa != nkx-1:
#       print(str(zarray[alfa][beta])+str(','), end = '')
#     else:
#       print(str(zarray[alfa][beta]), end = '')
#   if beta != nky-1:
#     print('},')
#   else:
#     print('}', end = '')
# print('};\n')


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

  #ax.plot_trisurf(xarray, yarray, zarray, linewidth=0.2, antialiased=True)

plt.show()




#    sys.exit("Stop")

# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))

