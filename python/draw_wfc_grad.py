# This program is to read and draw wfc and grad wfc

# This is to include maths
import math
import re
import numpy as np
# This is to draw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d
# This is to make operations in the shell
import sys
import time
# This are the subroutines and functions
import load
# This to make parallel processing
import joblib

memory = joblib.Memory(cachedir='/tmp/example/',verbose=0)

@memory.cache
def indices3(i,j,l):
  index = str(i)+' '+str(j)+' '+str(l)
  return index

@memory.cache
def indices2(i,j):
  index = str(i)+' '+str(j)
  return index


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

occ = load.readoccupancies(wfcdirectory,nbnd)    # {'58 6': 0.0, '25 8': 0.0, ...

apontador = load.readapontador(wfcdirectory,nbnd) # {'58 6': 8, '25 8': 8, '39 2': 3, ...

signaled = [line.rstrip('\n') for line in open(wfcdirectory+'/signaled','r')] # Reads the k-point,band signaled
#print(signaled)                                     # ['9 3', '9 4', '31 5']

ban = load.readbandas(wfcdirectory,nbnd)             # kpoint,newband-> old band

eigenvalues = {}                                     # Eigenvalues of the ordered states !!! not sure
for ki in range(nks):
  for bandas in range(1,nbnd+1):
    indx = indices2(ki,bandas)
    index = indices2(ki,apontador[indx])
    eigenvalues[index] = enerBands[indx]             # {'25 8': 0.336136887984, '39 2': -0.455816584941, ...
#    print(indx,eigenvalues[index],enerBands[index])
#print(eigenvalues)



print(' Start reading dump files')
wfcpos = joblib.load('./wfcpos3.gz')
wfcgra = joblib.load('./wfcgra3.gz')
#bandasr = joblib.load('./bandasr.gz')
#bandasg = joblib.load('./bandasg.gz')

print(' Finished reading data')
sys.stdout.flush()
pos,grax,gray = 0.0,0.0,0.0

for i in range(1,45503):
  pos += np.real(wfcpos[i])
  grax += np.real(wfcgra[i][0])
  gray += np.real(wfcgra[i][1])

#print(pos)
#print(grax)
#print(gray)

ponto = 10000
#print(np.real(wfcgra[ponto][0]))
#print(np.real(wfcgra[ponto][1]))
#print(np.real(wfcpos[ponto]))
a=np.real(wfcpos[ponto])
print(a.shape)


#print(wfcgra[0])
X, Y = np.meshgrid(np.arange(0, numero_ky), np.arange(0, numero_kx))
#print(X,Y)
print(X.shape)
print(Y.shape)

fig1, ax1 = plt.subplots()
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
#ax1.quiver(np.real(wfcgra[ponto][0]),np.real(wfcgra[ponto][1]), units='x',width=0.042,scale=1 / 1)
#ax.plot_wireframe(X,Y,np.real(wfcpos[ponto]))

ax1.quiver(grax,gray, units='x',width=0.042,scale=1 / 1)
ax.plot_wireframe(X,Y,pos)

plt.show()
# Cleaning
memory.clear(warn=False)

#sys.exit("Stop")



