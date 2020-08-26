###################################################################################
# This program runs a set of dft calculations, and prepares for further processing
###################################################################################
#
# This is to include maths
import numpy as np

# This is to make operations in the shell
import os
import sys
import time
import xml.etree.ElementTree as ET

# This are the subroutines and functions
import contatempo
import dft
from headerfooter import header,footer
from parser import parser

header('PREPROCESSING',time.asctime())

starttime = time.time()                         # Starts counting time

if len(sys.argv)!=2:
  print(' *!*!*!*!*!*!*!*!*!*!*!')
  print(' ERROR in number of arguments. No input file.')
  print(' *!*!*!*!*!*!*!*!*!*!*!')
  print()
  sys.exit("Stop")

print(' Reading from input file:', sys.argv[1])

# input file has to have these minimum parameters:
#     origin of k-points - k0
#     number of k-points in each direction - nkx, nky, nkz
#     step of the k-points - step
#     number of bands - nbnd
# and can have these other, that have defaults:
#     number of processors - np = 1
#     dft directory - dftdirectory = './dft/'
#     name_scf = 'scf'
#     name_nscf = 'nscf'
#     wfcdirectory = './wfc/' 

# Defaults:
npr = 1
dftdirectory = './dft/'
name_scf = 'scf'
name_nscf = 'nscf'
wfcdirectory = './wfc/'

# Open input file for the run
with open(sys.argv[1],'r') as inputfile:
  inputvar = inputfile.read().split("\n")
inputfile.close()

# Read that from input file
for i in inputvar:
  ii = i.split()
  if len(ii) == 0:
    continue
  if ii[0] == 'k0':
    k0 = [float(ii[1]),float(ii[2]),float(ii[3])]
  if ii[0] == 'nkx':
    nkx = int(ii[1])
  if ii[0] == 'nky':
    nky = int(ii[1])
  if ii[0] == 'nkz':
    nkz = int(ii[1])
  if ii[0] == 'step':
    step = float(ii[1])
  if ii[0] == 'nbnd':
    nbnd = int(ii[1])
  if ii[0] == 'npr':
    npr = int(ii[1])
  if ii[0] == 'dftdirectory':
    dftdirectory = ii[1]
  if ii[0] == 'name_scf':
    name_scf = ii[1]
  if ii[0] == 'name_nscf':
    name_nscf = ii[1]
  if ii[0] == 'wfcdirectory':
    wfcdirectory = ii[1]

print(' Number of bands in the nscf calculation nbnd:',str(nbnd))
print(' Starting k-point of the mesh k0:',str(k0))
print(' Number of points in the mesh ',str(nkx),str(nky),str(nkz))
print(' Step of the k-point mesh ',str(step))
print()
print(' Will run in',npr,' processors')
print(' The DFT files will be on:',dftdirectory)
print(' Name of scf file:',name_scf)
print(' Name od nscf file:',name_nscf)
print(' Name of directory for wfc:',wfcdirectory)
print()
print(' Finished reading input file')
print()

sys.stdout.flush()

if npr == 1:
  mpi = ''
else:
  mpi = 'mpirun -np ' + str(npr) + ' '

# Runs scf calculation  ** DFT
dft.scf(mpi,dftdirectory,name_scf)                 
# Creates template for nscf calculation  ** DFT
nscf = dft.template(dftdirectory,name_scf)         

nscfkpoints = ''                                # to append to nscf file
nks = nkx*nky*nkz                               # total number of k-points
nk = 0                                          # count the k-points
kpoints = np.zeros((nks,3), dtype=float)
for l in range(nkz):
  for j in range(nky):
    for i in range(nkx):
      k1, k2, k3 = round(k0[0] + step*i,8), round(k0[1] + step*j,8), round(k0[2] + step*l,8)
      kkk = "{:.7f}".format(k1) + '  ' + "{:.7f}".format(k2) + '  ' + "{:.7f}".format(k3)
      nscfkpoints = nscfkpoints + kkk + '  1\n'
      kpoints[nk,0] = k1
      kpoints[nk,1] = k2
      kpoints[nk,2] = k3
      nk = nk + 1

# Runs nscf calculations for all k-points  ** DFT **
dft.nscf(mpi,dftdirectory,name_nscf,nscf,nks,nscfkpoints,nbnd)  
sys.stdout.flush()

print(' Extracting data from DFT calculations')
prefix = parser('prefix',dftdirectory+name_nscf+'.in')
print('  DFT prefix:',prefix)
outdir = parser('outdir',dftdirectory+name_nscf+'.in')
print('  DFT outdir:',outdir)
dftdatafile = outdir+prefix+'.xml'
print('  DFT data file:',dftdatafile)
print()

tree = ET.parse(dftdatafile)
root = tree.getroot()

#for child in root[3]:
#  print(child.tag, child.attrib)
#  for child1 in child:
#    print(' ',child1.tag,child1.attrib)
#    for child2 in child1:
#      print('   ',child2.tag,child2.attrib)
#  print()

print(' Lattice vectors in units of a0 (bohr)')
for it in root[3][3][1].iter('a1'):
  a1 = np.array(list(map(float,it.text.split())))
print('  a1:',a1)
for it in root[3][3][1].iter('a2'):
  a2 = np.array(list(map(float,it.text.split())))
print('  a2:',a2)
for it in root[3][3][1].iter('a3'):
  a3 = np.array(list(map(float,it.text.split())))
print('  a3:',a3)
print()

print(' Reciprocal lattice vectors in units of 2pi/a0 (2pi/bohr)')
for it in root[3][5][9].iter('b1'):
  b1 = np.array(list(map(float,it.text.split())))
print('  b1:',b1)
for it in root[3][5][9].iter('b2'):
  b2 = np.array(list(map(float,it.text.split())))
print('  b2:',b2)
for it in root[3][5][9].iter('b3'):
  b3 = np.array(list(map(float,it.text.split())))
print('  b3:',b3)
print()

print(' Number of points in real space in each direction')
nr1 = int(root[3][5][3].attrib["nr1"])
nr2 = int(root[3][5][3].attrib["nr2"])
nr3 = int(root[3][5][3].attrib["nr3"])
nr = nr1*nr2*nr3
print('  nr1:',nr1)
print('  nr2:',nr2)
print('  nr3:',nr3)
print('  nr:',nr)
print()

nbnd = int(root[3][9][3].text)
print(' Number of bands in the DFT calculation: ',nbnd)
nks = int(root[3][9][10].text)
print(' Number of k-points in the DFT calculation: ',nks)
print()

#for child in root[3][9]:
#  print(child.tag, child.attrib,child.text)
#  for child1 in child:
#    print(' ',child1.tag,child1.attrib,child1.text)


eigenval = []
for it in root[3][9].iter('eigenvalues'):
  eigenval.append(list(map(float,it.text.split())))
eigenvalues = 2*np.array(eigenval)
#print(eigenvalues)

occupat = []
for it in root[3][9].iter('occupations'):
  occupat.append(list(map(float,it.text.split())))
occupations = np.array(occupat)
#print(occupations)

try:
  berrypath = str(os.environ['BERRYPATH'])
except KeyError:
  berrypath = str(os.path.dirname(os.path.dirname(__file__)))
print(' Path of BERRY files',berrypath)
print()

count = 0
r = np.zeros((nr,3), dtype=float)
for l in range(nr3):
  for k in range(nr2):
    for i in range(nr1):
      r[count] = a1*i/nr1 + a2*j/nr2 + a3*l/nr3
      count += 1

phase = np.zeros((nr,nks),dtype=complex)
for i in range(nr):
  for nk in range(nks):
    phase[i,nk] = np.exp(1j*(kpoints[nk,0]*r[i,0]\
                           + kpoints[nk,1]*r[i,1]\
                           + kpoints[nk,2]*r[i,2]))

with open('phase.npy','wb') as ph:
  np.save(ph,phase)
ph.closed
print(' Phases saved to file phase.npy')

neig = np.full((nks,4),-1,dtype=int)
nk = -1
with open('neighbors.dat','w') as nei:
  for j in range(nky):
    for i in range(nkx):
      nk = nk + 1
      if i == 0:
        n0 = -1
      else:
        n0 = nk - 1
      if j == 0:
        n1 = -1
      else:
        n1 = nk - nkx
      if i == nkx-1:
        n2 = -1
      else:
        n2 = nk + 1
      if j == nky-1:
        n3 = -1
      else:
        n3 = nk + nkx
      nei.write(str(nk)+'  '+str(n0)+'  '+str(n1)+'  '+str(n2)+'  '+str(n3)+'\n')
      neig[nk,0] = n0 
      neig[nk,1] = n1 
      neig[nk,2] = n2 
      neig[nk,3] = n3 
nei.closed
print(' Neighbors saved to file neighbors.dat')
with open('neighbors.npy','wb') as nnn:
  np.save(nnn,neig)
nnn.closed
print(' Neighbors saved to file neighbors.npy')



# Save eigenvalues to file (in Ha)
with open('eigenvalues.npy', 'wb') as f:
  np.save(f,eigenvalues)
f.closed
print(' Eigenvalues saved to file eigenvalues.npy (Ry)')


# Save occupations to file
with open('occupations.npy', 'wb') as f:
  np.save(f,occupations)
f.closed
print(' Occupations saved to file occupations.npy')

# Save positions to file
with open('positions.npy', 'wb') as f:
  np.save(f,r)
f.close()
print(' Positions saved to file positions.npy (bohr)')

# Save kpoints to file
with open('kpoints.npy', 'wb') as f:
  np.save(f,kpoints)
f.close()
print(' kpoints saved to file kpoints.npy (2pi/bohr)')

# Save data to file 'datafile.npy'
with open('datafile.npy', 'wb') as f:
  np.save(f,k0)             # Initial k-point
  np.save(f,nkx)            # Number of k-points in the x direction
  np.save(f,nky)            # Number of k-points in the y direction
  np.save(f,nkz)            # Number of k-points in the z direction
  np.save(f,nks)            # Total number of k-points
  np.save(f,step)           # Step between k-points
  np.save(f,npr)            # Number of processors for the run
  np.save(f,dftdirectory)   # Directory of DFT files
  np.save(f,name_scf)       # Name of scf file (without suffix)
  np.save(f,name_nscf)      # Name of nscf file (without suffix)
  np.save(f,wfcdirectory)   # Directory for the wfc files
  np.save(f,prefix)         # Prefix of the DFT QE calculations
  np.save(f,outdir)         # Directory for DFT saved files
  np.save(f,dftdatafile)    # Path to DFT file with data of the run
  np.save(f,a1)             # First lattice vector in real space
  np.save(f,a2)             # Second lattice vector in real space
  np.save(f,a3)             # Third lattice vector in real space
  np.save(f,b1)             # First lattice vector in reciprocal space
  np.save(f,b2)             # Second lattice vector in reciprocal space
  np.save(f,b3)             # Third lattice vector in reciprocal space
  np.save(f,nr1)            # Number of points of wfc in real space x direction
  np.save(f,nr2)            # Number of points of wfc in real space y direction
  np.save(f,nr3)            # Number of points of wfc in real space z direction
  np.save(f,nr)             # Total number of points of wfc in real space
  np.save(f,nbnd)           # Number of bands
  np.save(f,berrypath)      # Path of BERRY files
f.close()
print(' Data saved to file datafile.npy')

print()
nk = -1
sep = ' '
siz = nkx*nky

# Output the list of k-points in a convenient way
print('         | y  x ->')
for j in range(nky):
  lin = ''
  print()
  for i in range(nkx):
    nk = nk + 1
    if nk < 10:
      lin += sep+sep+sep+sep+str(nk)
    elif nk > 9 and nk < 100:
      lin += sep+sep+sep+str(nk)
    elif nk > 99 and nk < 1000:
      lin += sep+sep+str(nk)
    elif nk > 999 and nk < 10000:
      lin += sep+str(nk)
  print(lin)

# Finished
endtime = time.time()

footer(contatempo.tempo(starttime,endtime))



