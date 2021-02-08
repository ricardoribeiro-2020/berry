# This function receives as input a QE file and a keyword, and extracts the corresponding value

# This is to make operations in the shell
import os
import sys

def parser(keyword, qefile):

  strings = ['title','calculation','wf_collect','outdir','pseudo_dir','prefix','occupations','smearing']
  numbers = ['ibrav','nat','ntyp','nbnd']
  fnumbers = ['ecutwfc','degauss']
  if keyword in strings:
    string = os.popen('grep ' + keyword + ' ' + qefile).read().split()[2]
    return string.strip('\'')
  elif keyword in numbers:
    number = int(re.findall(r'\d+',os.popen('grep ' + keyword + ' ' + qefile).read())[0])
    return number
  elif keyword in fnumbers:
    number = float(re.findall(r'\d+\.\d+',os.popen('grep ' + keyword + ' ' + qefile).read())[0])
    return number
  elif keyword == 'CELL_PARAMETERS':
    lines = os.popen('grep -A3 ' + keyword + ' ' + qefile).read()
    cell = re.findall(r'\d+\.\d+',lines)
    print(cell[0:3])
    print(cell[3:6])
    print(cell[6:9])
  elif keyword == 'ATOMIC_SPECIES':
    spe = os.popen('grep ntyp ' + qefile).read()
    nspecies = int(re.findall(r'\d+',spe)[0])
    speci = os.popen('grep -A' + str(nspecies)+ ' ' + keyword + ' ' + qefile).read()
    species = speci.split()
    del species[0]
    for i in range(0,nspecies):
      print(i,species[3*i:3*i+3])
  elif keyword == 'ATOMIC_POSITIONS':
    nat = os.popen('grep nat ' + qefile).read()
    nats = int(re.findall(r'\d+',nat)[0])
    atom = os.popen('grep -A' + str(nats)+ ' ' + keyword + ' ' + qefile).read()
    atoms = atom.split()
    del atoms[0]
    del atoms[0]
    for i in range(0,nats):
      print(i,atoms[7*i:7*i+7])
  elif keyword == 'K_POINTS':
    kpoint = os.popen('grep -A2 ' + keyword + ' ' + qefile).read()
    print(kpoint.split()[-6:])
  return

