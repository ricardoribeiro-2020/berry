# This is to include maths
import math
import re
# This is to make operations in the shell
import os
import sys

def trimmedFile():
  lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  conta = lineList.index('# name: unkr')
  with open('wfcdata.dat','w') as wfcdata:
    for i in range(conta):
      wfcdata.write(lineList[i]+'\n')
  wfcdata.closed
  return





def FermiEnergy():
  verifica = os.popen("test -f wfcdata.dat;echo $?").read()            #if 1\n does no exist if 0\n exists
  if verifica == '1\n':
    lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  else:
    lineList = [line.rstrip('\n') for line in open('wfcdata.dat','r')] # Reads file wfck2r.dat, puts in list

  efermi = float(lineList[lineList.index('# name: efermi') + 2])       # Finds Fermi energy
  #print('efermi ',efermi)
  return efermi





def kPointCoordinates():
  verifica = os.popen("test -f wfcdata.dat;echo $?").read()            #if 1\n does no exist if 0\n exists
  if verifica == '1\n':
    lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  else:
    lineList = [line.rstrip('\n') for line in open('wfcdata.dat','r')] # Reads file wfck2r.dat, puts in list
  xk = []
  initxk = int(lineList.index('# name: xk')) + 4                       # Finds k-point coordinates, puts in list
  xk.append(float(lineList[initxk]))
  xk.append(float(lineList[initxk+1]))
  xk.append(float(lineList[initxk+2]))
  #print('xk ',xk)
  return xk




def numberBands():
  verifica = os.popen("test -f wfcdata.dat;echo $?").read()            #if 1\n does no exist if 0\n exists
  if verifica == '1\n':
    lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  else:
    lineList = [line.rstrip('\n') for line in open('wfcdata.dat','r')] # Reads file wfck2r.dat, puts in list
  nbands = int(lineList[lineList.index('# name: nbands') + 2])         # Finds number of bands in file
  #print('nbands ',nbands)
  return nbands





def eigenvalues():
  verifica = os.popen("test -f wfcdata.dat;echo $?").read()            #if 1\n does no exist if 0\n exists
  if verifica == '1\n':
    lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  else:
    lineList = [line.rstrip('\n') for line in open('wfcdata.dat','r')] # Reads file wfck2r.dat, puts in list
  nbands = int(lineList[lineList.index('# name: nbands') + 2])         # Finds number of bands in file
  eig = []
  inibands = int(lineList.index('# name: eigs')) + 4                   # Reads eigenvalues and puts in list eig
#  with open("energias","w") as energias:
  for b in range(nbands):
    eig.append(float(lineList[inibands+b]))
#      energias.write(lineList[inibands+b] + '\n')                     # Writes eigenvalues to file 'energias'
#  energias.closed
#  print(eig)
  return eig





# Returns occupancies
def occupancies():
  verifica = os.popen("test -f wfcdata.dat;echo $?").read()            #if 1\n does no exist if 0\n exists
  if verifica == '1\n':
    lineList = [line.rstrip('\n') for line in open('wfck2r.mat','r')]  # Reads file wfck2r.mat, puts in list
  else:
    lineList = [line.rstrip('\n') for line in open('wfcdata.dat','r')] # Reads file wfck2r.dat, puts in list
  nbands = int(lineList[lineList.index('# name: nbands') + 2])         # Finds number of bands in file
  occ = []
  inioccup = int(lineList.index('# name: occup')) + 4                  # Reads occupancies and puts in list occ
#  with open("ocupacoes","w") as occup:
  for b in range(nbands):
    occ.append(float(lineList[inioccup+b]))
#      occup.write(lineList[inioccup+b] + '\n')                        # Writes to file 'ocupacoes'
#  occup.closed
  return occ




