# This program simply converts the 'apontador' file to 'bandas' file
# apontador indicates for each original kpoint, band -> new band
# bandas indicates for each new band,kpoint -> original band

# This is to include maths
import math
import numpy as np
import ast

wfcdirectory = 'wfc'

with open(wfcdirectory+'/k_points','r') as ks:
  kpoints = ks.read().split("\n")               # List of k-points
ks.closed                                       # ['0.00000  0.00000  0.00000   ', '0.01000  0.00000  0.00000   ',...
del kpoints[-1]
#print(kpoints)
nks = len(kpoints)                              # number of k-points in the calculation: nks

enerBands = {}
with open(wfcdirectory+'/eigenvalues','r') as ei:   # Reads eigenvalues from file
  line = ei.readline()                          # Reads line
  nbnd = len(line.split()) - 1                  # Number of bands in the DFT calculation: nbnd
  #print(nbnd)
  while line:                                   # for every line
    bands = line.split()                        # split into bands, first column is k-index
    for i in range(nbnd):
      indx = bands[0] + ' ' + str(i+1)          # For each k and band
      enerBands[indx] = float(bands[i+1])       # {'58 6': 0.28961845029, '25 8': 0.336136887984, ...
    line = ei.readline()
ei.closed
#print(enerBands)

occ = {}
with open(wfcdirectory+'/occupancies','r') as occup: # Reads band occupancy
  line = occup.readline()
  while line:
    bands = line.split()                             # split into bands, first column is k-index
    for i in range(nbnd):
      indx = bands[0] + ' ' + str(i+1)               # For each k and band
      occ[indx] = float(float(bands[i+1]))           # {'58 6': 0.0, '25 8': 0.0, ...
    line = occup.readline()
occup.closed
#print(occ)


apontador = {}
with open(wfcdirectory+'/apontador','r') as aa:      # Reads pointers of bands from file
  line = aa.readline()
  while line:
    apont = line.split()
    for i in range(nbnd):
      indx = apont[0] + ' ' + apont[1]
      apontador[indx] = int(apont[2])                # {'58 6': 8, '25 8': 8, '39 2': 3, ...
    line = aa.readline()
aa.closed
#print(apontador)
napont = len(apontador)

with open(wfcdirectory+'/bandas','w') as ba:
  for l in range(1,nbnd+1):
    for i in range(nks):
      for j in range(1,nbnd+1):
        indx = str(i) + ' ' + str(j)
        if indx in apontador.keys():
          if l==apontador[indx]:
            alfa = str(l) + ' ' + str(i) + ' ' + str(j)
            ba.write(alfa+'\n')
ba.closed




