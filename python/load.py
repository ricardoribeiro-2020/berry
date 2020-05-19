# This set of programs read data from the files
# This is to include maths
import math
import re
import numpy as np
# This are the subroutines and functions
from auxiliary import indices2, indices3
# This is to make operations in the shell
import sys
# This to make parallel processing
import joblib

### readkindex(wfcdirectory):
### readenergias1(wfcdirectory):
### readenergias(wfcdirectory):
### readoccupancies(wfcdirectory,nbnd):
### readapontador(wfcdirectory,nbnd):                  # Ignores fourth column
### readsignaled(wfcdirectory,nbnd):
### readbandas(wfcdirectory,nbnd):
### readwfc(wfcdirectory,kp,banda,col='a'):

def readkindex(wfcdirectory):
  kindex = {}
  with open(wfcdirectory+'/kindex','r') as ki:
    knumber = ki.readline().split(' ')
    numero_kx = int(knumber[0])                  # Number of k-points in each direction
    numero_ky = int(knumber[1])
    numero_kz = int(knumber[2])
#    print(numero_kx,numero_ky,numero_kz)
    for l in range(numero_kz):
      for j in range(numero_ky):
        for i in range(numero_kx):
          index = indices3(i,j,l)                # Dictionary to translate kx,ky,kz -> nk
          linha = ki.readline().split('  ')      # {'7 7 0': 63, '7 4 0': 39, '4 5 0': 44,...
          kindex[index] = int(linha[1].replace("\n",""))
#          print(index,kindex[index])
  ki.closed
#  print(kindex)

  with open(wfcdirectory+'/k_points','r') as ks:
    kpoints = ks.read().split("\n")               # List of k-points
  ks.closed                                       # ['0.00000  0.00000  0.00000   ', '0.01000  0.00000  0.00000   ',...
  del kpoints[-1]
#  print(kpoints)
  nks = len(kpoints)                              # number of k-points in the calculation: nks

  kpointx,kpointy,kpointz = {},{},{}              # kx, ky, kz as function of nk
  for ii in range(nks):
    i = kpoints[ii].split()
    kpointx[ii] = float(i[0])                     # {0: 0.0, 1: 0.01, 2: 0.02, ...
    kpointy[ii] = float(i[1])
    kpointz[ii] = float(i[2])
  dk = kpointx[1] - kpointx[0]                    # Defines the step for gradient calculation dk
#  print(kpointx)

  return numero_kx,numero_ky,numero_kz,kindex,nks,kpointx,kpointy,kpointz,dk




def readenergias1(wfcdirectory):
  enerBands = {}
  with open(wfcdirectory+'/eigenvalues','r') as ei:   # Reads eigenvalues from file
    line = ei.readline()                          # Reads line
    nbnd = len(line.split()) - 1                  # Number of bands in the DFT calculation: nbnd
    nks = 0
    #print(nbnd)
    while line:                                   # for every line
      nks = nks + 1
      bands = line.split()                        # split into bands, first column is k-index
      for i in range(nbnd):
        indx = indices2(int(bands[0]),i+1)          # For each k and band
        enerBands[indx] = float(bands[i+1])       # {'58 6': 0.28961845029, '25 8': 0.336136887984, ...
      line = ei.readline()
  ei.closed

  return nbnd,enerBands





def readenergias(wfcdirectory):
  enerBands = {}
  with open(wfcdirectory+'/eigenvalues','r') as ei:   # Reads eigenvalues from file
    line = ei.readline()                          # Reads line
    nbnd = len(line.split()) - 1                  # Number of bands in the DFT calculation: nbnd
    nks = 0
    #print(nbnd)
    while line:                                   # for every line
      nks = nks + 1
      bands = line.split()                        # split into bands, first column is k-index
      for i in range(nbnd):
        indx = indices2(int(bands[0]),i+1)          # For each k and band
        enerBands[indx] = float(bands[i+1])       # {'58 6': 0.28961845029, '25 8': 0.336136887984, ...
      line = ei.readline()
  ei.closed

  apontador = readapontador(wfcdirectory,nbnd)
  eigenvalues = {}                           # Dictionary with index k,band -> eigenvalue
  for i in range(nks):                       # These are the corrected values of the bands
    for j in range(1,nbnd+1):
      ss = str(i)+' '+str(j)
      if ss in apontador.keys():
        index = str(i)+' '+str(apontador[ss])
        if index in enerBands.keys():
          eigenvalues[index] = float(enerBands[ss])
        else:
          eigenvalues[index] = 0.0

  return nbnd,enerBands,nks,eigenvalues




def readoccupancies(wfcdirectory,nbnd):
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
#  print(occ)

  return occ





def readapontador(wfcdirectory,nbnd):                  # Ignores fourth column
  apontador = {}
  with open(wfcdirectory+'/apontador','r') as aa:      # Reads pointers of bands from file
    line = aa.readline()
    while line:
      apont = line.split()
      for i in range(nbnd):
        indx = apont[0] + ' ' + apont[1]
        if indx not in apontador.keys():
          apontador[indx] = int(apont[2])                # {'58 6': 8, '25 8': 8, '39 2': 3, ...
      line = aa.readline()
  aa.closed
#  print(apontador)

  return apontador

def readsignaled(wfcdirectory,nbnd):
  apontador = {}
  signaled  = {}
  with open(wfcdirectory+'/apontador','r') as aa:      # Reads pointers of bands from file
    line = aa.readline()
    while line:
      apont = line.split()
      for i in range(nbnd):
        indx = apont[0] + ' ' + apont[1]
        if indx not in apontador.keys():
          apontador[indx] = int(apont[2])                # {'58 6': 8, '25 8': 8, '39 2': 3, ...
          signaled[indx] = int(apont[3]) 
      line = aa.readline()
  aa.closed
#  print(apontador)

  return apontador, signaled



def readbandas(wfcdirectory,nbnd):
  ban = {}
  with open(wfcdirectory+'/bandas','r') as ba:         # Reads pointers from file bandas
    line = ba.readline()
    while line:
      bbb = line.split()
      for i in range(nbnd):
        indx = indices2(int(bbb[1]),int(bbb[0]))                   # kpoint,newband-> old band
        ban[indx] = int(bbb[2])
      line = ba.readline()
  ba.closed
#  print(ban)
  return ban





def readwfc(wfcdirectory,kp,banda,col='a'):
  if col == 'ab':
    fich = wfcdirectory+"/k000"+str(kp)+"b000"+str(banda)+".wfc1"
  else:
    fich = wfcdirectory+"/k000"+str(kp)+"b000"+str(banda)+".wfc"
  lineList = [line.rstrip('\n') for line in open(fich,'r')]  # Reads wfc file

  if col == 'a' or col == 'ab':
    wfc = lineList
  elif col == 'w':
    wfc = []
    for i in lineList:
      x = i.split()
      y = x[3]+'  '+x[4]+'  '+x[5]
      wfc.append(y)
  elif col == 'r':
    wfc = []
    for i in lineList:
      x = i.split()
      y = x[4]
      wfc.append(y)
  elif col == 'i':
    wfc = []
    for i in lineList:
      x = i.split()
      y = x[5]
      wfc.append(y)

  return wfc    # Returns a list








