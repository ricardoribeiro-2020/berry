# These routines run the DFT calculations

import readwfck2r
# This is to make operations in the shell
import os
import sys

# SCF calculation
def scf(mpi,directory,name_scf):

  scffile = directory + name_scf
  verifica = os.popen("test -f " + scffile + ".out;echo $?").read()   #if 1\n does no exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the scf calculation
    print(' Running scf calculation')
    comando = mpi + "pw.x -i " + scffile +".in" + " > " + scffile + ".out"
    print(' Command: '+comando+'\n')
    os.system(comando)
  return




# Creates template for nscf calculation
def template(directory,name_scf):

  # opens the template file from which it will do nscf calculations
  scffile = directory + name_scf +".in"
  with open(scffile,'r') as templa:
    original = templa.read()
  templa.closed
  nscf1 = original.replace('automatic','tpiba').replace('scf','nscf')

  nscf2 = nscf1.split('\n')
  for i in range(1,5):                   # Clean empty lines in the end of the file
    if nscf2[-1] == '':
      del nscf2[-1]

  del nscf2[-1]                          # Delete SCF k points

  nscf = '\n'.join(nscf2)                # Build the template nscf file

  return nscf




# NSCF calculation ****************************************************
def nscf(mpi,directory,name_nscf,nscf,nkps,kpoints,nbands):
  nscffile = directory + name_nscf
  verifica = os.popen("test -f " + nscffile + ".out;echo $?").read()   #if 1\n does no exist if 0\n exists
  
  if verifica == '1\n':  # If output file does not exist, run the nscf calculation

    lines = nscf.split('\n')
    subst = ''
    for line in lines:
      if line.find('nbnd') >= 0:
        rep = line.split()
        subst = subst + line.find('nbnd')*' '+'nbnd = ' + str(nbands) + ' ,\n'
      else:
        subst = subst + line.rstrip() + '\n'

    nscf = subst

    with open(nscffile+'.in','w') as output:
      output.write(nscf + str(nkps) + '\n' + kpoints)
    output.closed

#    sys.exit("Stop")

    print(' Running nscf calculation')
    comando = mpi + "pw.x -i " + nscffile +".in" + " > " + nscffile + ".out"
    print(' Command: '+comando+'\n')
    os.system(comando)
  return





# wfck2r calculation and wavefunctions extraction *********************
def wfck2r(directory,nk1,wfcdirectory,create=True,limite=400):
  print(' Extracting wavefunctions; limite = ',limite,' nk = ',nk1)
  sys.stdout.flush()
  comando = "&inputpp  prefix = 'bn' , outdir = '"+directory+"out/' , first_k = "+str(nk1+1)+", last_k = "+str(nk1+1)+",loctave=.true., /"
  if create:
    os.system('echo "'+comando+'"|wfck2r.x > '+directory+'tmp')
    #os.system('ls -l wfck2r.mat')

    comando2 = "&input limite = "+str(limite)+", nk1 = "+str(nk1)+", wfcdirectory = '"+wfcdirectory+"',/"
    os.system('echo "'+comando2+'"|./extractwfc.x')

  readwfck2r.trimmedFile()                        # cuts data from the rest of file
  #print(comando)

  return






